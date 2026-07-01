from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = ROOT_DIR / "camera_pose_calibration.json"

POINT_TRANSFORMS = {
    "left": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=float),
    "right": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
}

RIGHT_HANDED_FIXES = {
    "flip_x": np.diag([-1.0, 1.0, 1.0]),
    "flip_y": np.diag([1.0, -1.0, 1.0]),
    "flip_z": np.diag([1.0, 1.0, -1.0]),
}

CAMERA_NAME_MAP = {
    "left": "cam1",
    "right": "cam2",
}


@dataclass
class MujocoPoseCandidate:
    name: str
    transform_matrix: list[list[float]]
    score: float
    rotation_matrix: list[list[float]]
    quat_wxyz: list[float]


@dataclass
class MujocoCameraPose:
    camera_name: str
    selected_candidate: str
    point_transform_matrix: list[list[float]]
    rotation_transform_matrix: list[list[float]]
    fixed_rotation_matrix: list[list[float]] | None
    refinement_euler_xyz_deg: list[float]
    rotation_matrix: list[list[float]]
    quat_wxyz: list[float]
    candidates: list[dict[str, Any]]


def rotation_matrix_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


def _quat_wxyz_from_matrix(rotation: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _matrix_from_quat_wxyz(quat_wxyz: list[float]) -> np.ndarray:
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()


def _candidate_score(rotation_mj: np.ndarray) -> float:
    x_axis = rotation_mj[:, 0]
    y_axis = rotation_mj[:, 1]
    z_axis = rotation_mj[:, 2]
    return float(
        2.0 * max(y_axis[2], 0.0)
        + 2.0 * max(z_axis[2], 0.0)
        - 0.5 * abs(x_axis[2])
    )


def load_camera_pose_calibration(path: str | Path = DEFAULT_CALIBRATION_PATH) -> dict[str, Any]:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return {}
    return json.loads(calibration_path.read_text(encoding="utf-8"))


def _calibration_entry(calibration: dict[str, Any], camera_name: str) -> dict[str, Any]:
    entry = calibration.get(camera_name)
    if not isinstance(entry, dict):
        return {}
    return entry


def _heuristic_rotation_candidates(raw_rotation: np.ndarray, camera_name: str) -> list[MujocoPoseCandidate]:
    improper = POINT_TRANSFORMS[camera_name]
    candidates: list[MujocoPoseCandidate] = []
    for fix_name, fix in RIGHT_HANDED_FIXES.items():
        transform = improper @ fix
        if not np.isclose(np.linalg.det(transform), 1.0, atol=1e-6):
            continue
        rotation_mj = transform @ raw_rotation
        score = _candidate_score(rotation_mj)
        candidates.append(
            MujocoPoseCandidate(
                name=fix_name,
                transform_matrix=transform.tolist(),
                score=score,
                rotation_matrix=rotation_mj.tolist(),
                quat_wxyz=_quat_wxyz_from_matrix(rotation_mj),
            )
        )
    return candidates


def build_calibration_from_reference_scene(
    reference_scene_path: str | Path,
    raw_euler_deg_by_camera: dict[str, list[float]],
) -> dict[str, Any]:
    root = ET.fromstring(Path(reference_scene_path).read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in {reference_scene_path}")

    calibration: dict[str, Any] = {
        "_meta": {
            "reference_scene": str(Path(reference_scene_path).resolve()),
        }
    }

    for camera_name, scene_camera_name in CAMERA_NAME_MAP.items():
        camera_node = next(
            (camera for camera in worldbody.findall("camera") if camera.get("name") == scene_camera_name),
            None,
        )
        if camera_node is None:
            raise ValueError(f"Camera {scene_camera_name} not found in {reference_scene_path}")

        quat_text = camera_node.get("quat")
        if not quat_text:
            raise ValueError(f"Camera {scene_camera_name} missing quat in {reference_scene_path}")

        reference_quat = [float(value) for value in quat_text.split()]
        raw_euler_deg = raw_euler_deg_by_camera[camera_name]
        raw_rotation = rotation_matrix_from_euler_xyz_deg(*raw_euler_deg)
        reference_rotation = _matrix_from_quat_wxyz(reference_quat)
        fixed_rotation = reference_rotation @ raw_rotation.T

        calibration[camera_name] = {
            "scene_camera_name": scene_camera_name,
            "raw_euler_xyz_deg": [float(value) for value in raw_euler_deg],
            "reference_quat_wxyz": reference_quat,
            "reference_rotation_matrix": reference_rotation.tolist(),
            "fixed_rotation_matrix": fixed_rotation.tolist(),
            "fixed_quat_wxyz": _quat_wxyz_from_matrix(fixed_rotation),
            "refinement_euler_xyz_deg": [0.0, 0.0, 0.0],
        }

    return calibration


def save_camera_pose_calibration(calibration: dict[str, Any], path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
    out_path = Path(path).resolve()
    out_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def convert_raw_rotation_to_mujoco(
    raw_rotation: np.ndarray,
    camera_name: str,
    calibration: dict[str, Any] | None = None,
) -> MujocoCameraPose:
    if camera_name not in POINT_TRANSFORMS:
        raise ValueError(f"Unsupported camera_name={camera_name!r}")

    if calibration is None:
        calibration = load_camera_pose_calibration()

    entry = _calibration_entry(calibration, camera_name)
    improper = POINT_TRANSFORMS[camera_name]

    if entry:
        fixed_rotation = np.asarray(entry["fixed_rotation_matrix"], dtype=float)
        refinement_euler = np.asarray(entry.get("refinement_euler_xyz_deg", [0.0, 0.0, 0.0]), dtype=float)
        refinement_rotation = Rotation.from_euler("xyz", refinement_euler, degrees=True).as_matrix()
        rotation_mj = refinement_rotation @ fixed_rotation @ raw_rotation
        candidate = MujocoPoseCandidate(
            name="reference_calibration",
            transform_matrix=improper.tolist(),
            score=0.0,
            rotation_matrix=rotation_mj.tolist(),
            quat_wxyz=_quat_wxyz_from_matrix(rotation_mj),
        )
        return MujocoCameraPose(
            camera_name=camera_name,
            selected_candidate="reference_calibration",
            point_transform_matrix=improper.tolist(),
            rotation_transform_matrix=improper.tolist(),
            fixed_rotation_matrix=fixed_rotation.tolist(),
            refinement_euler_xyz_deg=[float(value) for value in refinement_euler.tolist()],
            rotation_matrix=candidate.rotation_matrix,
            quat_wxyz=candidate.quat_wxyz,
            candidates=[asdict(candidate)],
        )

    candidates = _heuristic_rotation_candidates(raw_rotation, camera_name)
    if not candidates:
        raise ValueError(f"No valid MuJoCo rotation candidate generated for {camera_name}")
    selected = max(candidates, key=lambda item: item.score)
    return MujocoCameraPose(
        camera_name=camera_name,
        selected_candidate=selected.name,
        point_transform_matrix=improper.tolist(),
        rotation_transform_matrix=selected.transform_matrix,
        fixed_rotation_matrix=None,
        refinement_euler_xyz_deg=[0.0, 0.0, 0.0],
        rotation_matrix=selected.rotation_matrix,
        quat_wxyz=selected.quat_wxyz,
        candidates=[asdict(item) for item in candidates],
    )


def convert_raw_pose_payload(
    raw_pose: dict[str, Any],
    camera_name: str,
    calibration: dict[str, Any] | None = None,
) -> MujocoCameraPose:
    raw_rotation = np.asarray(raw_pose["rotation_matrix"], dtype=float)
    return convert_raw_rotation_to_mujoco(raw_rotation, camera_name, calibration=calibration)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a raw camera pose into a MuJoCo camera rotation.")
    parser.add_argument("--camera", choices=sorted(POINT_TRANSFORMS))
    parser.add_argument("--raw-pose-json", help="Path to a raw pose JSON file generated by camera_pose_ransac.py.")
    parser.add_argument("--euler-xyz-deg", help="Fallback input as rx,ry,rz in degrees.")
    parser.add_argument("--json-out", help="Optional output JSON path.")
    parser.add_argument("--calibration-json", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--build-calibration-from-scene", help="Reference scene XML used to build fixed camera calibration.")
    parser.add_argument("--left-euler-xyz-deg", default="-17.08,-40.16,96.59")
    parser.add_argument("--right-euler-xyz-deg", default="-97.480497,-1.078239,-0.070489")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.build_calibration_from_scene:
        left_euler = [float(value.strip()) for value in args.left_euler_xyz_deg.split(",")]
        right_euler = [float(value.strip()) for value in args.right_euler_xyz_deg.split(",")]
        calibration = build_calibration_from_reference_scene(
            args.build_calibration_from_scene,
            {"left": left_euler, "right": right_euler},
        )
        out_path = save_camera_pose_calibration(calibration, args.calibration_json)
        print(json.dumps({"saved_to": str(out_path), "calibration": calibration}, ensure_ascii=False, indent=2))
        return

    if not args.camera:
        raise SystemExit("请传入 --camera，或使用 --build-calibration-from-scene")

    calibration = load_camera_pose_calibration(args.calibration_json)
    if args.raw_pose_json:
        raw_pose = json.loads(Path(args.raw_pose_json).read_text(encoding="utf-8"))
        pose = convert_raw_pose_payload(raw_pose, args.camera, calibration=calibration)
    elif args.euler_xyz_deg:
        rx, ry, rz = [float(value.strip()) for value in args.euler_xyz_deg.split(",")]
        raw_rotation = rotation_matrix_from_euler_xyz_deg(rx, ry, rz)
        pose = convert_raw_rotation_to_mujoco(raw_rotation, args.camera, calibration=calibration)
    else:
        raise SystemExit("请传入 --raw-pose-json 或 --euler-xyz-deg")

    payload = json.dumps(asdict(pose), ensure_ascii=False, indent=2)
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
