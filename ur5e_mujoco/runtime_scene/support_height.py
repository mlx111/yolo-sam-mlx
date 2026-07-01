from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def _fmt(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _quat_wxyz_to_matrix(quat_wxyz: list[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion.")
    quat = quat / norm
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _parse_floats(raw: str | None, expected: int, name: str) -> list[float]:
    if raw is None:
        raise ValueError(f"Missing {name}.")
    values = [float(part) for part in raw.split()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values for {name}, got {len(values)}: {raw}")
    return values


def support_adjusted_position(
    *,
    pos_xyz: list[float],
    quat_wxyz: list[float],
    mesh_path: str | Path,
    support_z: float = 0.0,
    clearance: float = 0.001,
) -> tuple[list[float], dict[str, float | bool | str]]:
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    pos = np.asarray(pos_xyz, dtype=np.float64)
    rotation_world = _quat_wxyz_to_matrix(quat_wxyz)
    min_z = float((vertices @ rotation_world.T)[:, 2].min() + pos[2])
    target_min_z = float(support_z) + float(clearance)
    delta_z = max(0.0, target_min_z - min_z)
    adjusted = pos.copy()
    adjusted[2] += delta_z
    return adjusted.tolist(), {
        "mesh_path": str(Path(mesh_path).resolve()),
        "support_z": float(support_z),
        "clearance": float(clearance),
        "min_z_before": min_z,
        "target_min_z": target_min_z,
        "delta_z": float(delta_z),
        "adjusted": bool(delta_z > 0.0),
    }


def adjust_scene_support_heights(
    *,
    scene_path: str | Path,
    mesh_paths: dict[str, str | Path],
    body_by_object: dict[str, str],
    support_z: float = 0.0,
    clearance: float = 0.001,
) -> dict[str, dict[str, float | bool | str | list[float]]]:
    scene = Path(scene_path)
    tree = ET.parse(scene)
    root = tree.getroot()
    adjustments: dict[str, dict[str, float | bool | str | list[float]]] = {}

    for object_name, body_name in body_by_object.items():
        if object_name not in mesh_paths:
            continue
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        pos = _parse_floats(body.get("pos"), 3, f"{body_name}.pos")
        quat = _parse_floats(body.get("quat"), 4, f"{body_name}.quat")
        adjusted_pos, debug = support_adjusted_position(
            pos_xyz=pos,
            quat_wxyz=quat,
            mesh_path=mesh_paths[object_name],
            support_z=support_z,
            clearance=clearance,
        )
        if debug["adjusted"]:
            body.set("pos", _fmt(adjusted_pos))
        adjustments[object_name] = {
            **debug,
            "body": body_name,
            "pos_before": pos,
            "pos_after": adjusted_pos,
        }

    if any(bool(item["adjusted"]) for item in adjustments.values()):
        tree.write(scene, encoding="utf-8", xml_declaration=True)
    return adjustments
