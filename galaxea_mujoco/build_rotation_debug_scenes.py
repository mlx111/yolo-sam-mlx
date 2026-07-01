from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from build_runtime_scene_from_real_camera import _camera_to_target_mesh_quat
from dong2 import generate_scene


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT_DIR / "results" / "runtime_scenes" / "real_camera_runtime_scene.json"
DEFAULT_OUT_DIR = ROOT_DIR / "scence" / "rotation_debug"


def _normalize_object_name(name: str) -> str:
    return str(name).strip().lower().rstrip(".").replace(" ", "_")


def _quat_wxyz_to_matrix(quat: list[float]) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"Expected wxyz quaternion with 4 values, got {values}")
    values = values / max(float(np.linalg.norm(values)), 1e-12)
    return Rotation.from_quat([values[1], values[2], values[3], values[0]]).as_matrix()


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _variant_rotations(base_rotation: np.ndarray) -> dict[str, np.ndarray]:
    # 180 deg about camera/MuJoCo X: equivalent to flipping Y and Z.
    x180 = np.diag([1.0, -1.0, -1.0])
    y180 = np.diag([-1.0, 1.0, -1.0])
    z180 = np.diag([-1.0, -1.0, 1.0])
    current_x180 = base_rotation @ x180
    return {
        "identity": np.eye(3),
        "current": base_rotation,
        "current_T": base_rotation.T,
        "current_x180": current_x180,
        "x180_current": x180 @ base_rotation,
        "current_T_x180": base_rotation.T @ x180,
        "x180_current_T": x180 @ base_rotation.T,
        "world_x180_current_x180": x180 @ current_x180,
        "world_y180_current_x180": y180 @ current_x180,
        "world_z180_current_x180": z180 @ current_x180,
    }


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing runtime scene report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_rotation_debug_scenes(
    *,
    report_path: Path = DEFAULT_REPORT,
    out_dir: Path = DEFAULT_OUT_DIR,
    objects: list[str] | None = None,
) -> dict[str, Any]:
    report = _load_report(report_path)
    positions_raw = report.get("object_positions_m")
    if not isinstance(positions_raw, dict) or not positions_raw:
        raise ValueError(f"Report missing object_positions_m: {report_path}")

    selected_objects = [_normalize_object_name(name) for name in objects] if objects else None
    object_positions = {
        _normalize_object_name(name): [float(v) for v in pos]
        for name, pos in positions_raw.items()
        if selected_objects is None or _normalize_object_name(name) in selected_objects
    }
    if not object_positions:
        raise ValueError(f"No matching object positions. objects={objects}, available={sorted(positions_raw)}")

    target_coordinate_system = str(report.get("target_coordinate_system", "world"))
    report_mesh_quats = {
        _normalize_object_name(name): [float(v) for v in quat]
        for name, quat in dict(report.get("mesh_quats") or {}).items()
    }
    base_quat = next(iter(report_mesh_quats.values()), None)
    if base_quat is None:
        base_quat = _camera_to_target_mesh_quat(target_coordinate_system)
    base_rotation = _quat_wxyz_to_matrix(base_quat)

    out_dir.mkdir(parents=True, exist_ok=True)
    variants = _variant_rotations(base_rotation)
    outputs: dict[str, Any] = {}
    for variant_name, rotation in variants.items():
        mesh_quats = {name: _matrix_to_quat_wxyz(rotation) for name in object_positions}
        scene_out = out_dir / f"real_camera_runtime_{variant_name}.xml"
        generated = generate_scene(
            object_positions,
            scene_out=scene_out,
            mesh_quats=mesh_quats,
        )
        outputs[variant_name] = {
            "scene_out": str(generated.resolve()),
            "mesh_quats": mesh_quats,
            "view_command": f"python -m mujoco.viewer --mjcf {generated}",
        }

    summary = {
        "report_path": str(report_path.resolve()),
        "target_coordinate_system": target_coordinate_system,
        "objects": sorted(object_positions),
        "base_quat_wxyz": base_quat,
        "outputs": outputs,
    }
    summary_path = out_dir / "rotation_debug_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate runtime scene XMLs with rotation variants for visual debugging.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Runtime scene JSON report.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for debug XMLs.")
    parser.add_argument("--objects", nargs="*", default=None, help="Optional object names to include.")
    args = parser.parse_args()

    summary = build_rotation_debug_scenes(
        report_path=Path(args.report).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        objects=args.objects,
    )
    print(f"summary: {summary['summary_path']}")
    for name, item in summary["outputs"].items():
        print(f"{name}: {item['scene_out']}")


if __name__ == "__main__":
    main()
