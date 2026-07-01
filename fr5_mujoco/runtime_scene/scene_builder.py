"""Build UR5e MuJoCo runtime scenes from object poses.

This module is intentionally local to ``ur5e_mujoco`` so runtime scene
generation does not depend on root-level helper scripts or manipulator_grasp
scene templates.
"""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


UR5E_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = UR5E_ROOT / "assets"
DEFAULT_TEMPLATE = ASSET_ROOT / "scenes" / "scene2.xml"
DEFAULT_OUTPUT = UR5E_ROOT / "scene" / "scene.xml"
FRUIT_STL_DIR = ASSET_ROOT / "fruit" / "stl"

APPLE = "apple"
PEAR = "pear"


class SceneGenerationError(RuntimeError):
    """Raised when runtime scene generation cannot complete."""


def _ensure_position(pos: list[float] | tuple[float, float, float], object_name: str) -> list[float]:
    if len(pos) != 3:
        raise SceneGenerationError(f"{object_name} position must contain exactly 3 values.")
    coords = [float(pos[0]), float(pos[1]), float(pos[2])]
    coords[2] = max(0.0, coords[2])
    return coords


def _ensure_vector3(vec: list[float] | tuple[float, float, float], name: str) -> list[float]:
    if len(vec) != 3:
        raise SceneGenerationError(f"{name} position must contain exactly 3 values.")
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def _ensure_quat(quat: list[float] | tuple[float, float, float, float], name: str) -> list[float]:
    if len(quat) != 4:
        raise SceneGenerationError(f"{name} quaternion must contain exactly 4 values.")
    values = np.asarray([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-9:
        raise SceneGenerationError(f"{name} quaternion norm must be non-zero.")
    return (values / norm).tolist()


def _quat_to_matrix(quat: list[float] | tuple[float, float, float, float], name: str) -> np.ndarray:
    values = _ensure_quat(quat, name)
    return Rotation.from_quat([values[1], values[2], values[3], values[0]]).as_matrix()


def _compose_quats(
    object_quat: list[float] | tuple[float, float, float, float] | None,
    mesh_quat: list[float] | tuple[float, float, float, float] | None,
    name: str,
    default_quat: list[float],
) -> list[float]:
    base_rotation = _quat_to_matrix(object_quat or default_quat, f"{name}_object")
    combined = base_rotation if mesh_quat is None else base_rotation @ _quat_to_matrix(mesh_quat, f"{name}_mesh")
    quat_xyzw = Rotation.from_matrix(combined).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _template_text(template: Path) -> str:
    text = template.read_text(encoding="utf-8")
    return (
        text.replace("../universal_robots_ur5e/", "../universal_robots_ur5e/")
        .replace("../robotiq_2f85/", "../robotiq_2f85/")
        .replace("../table/", "../table/")
        .replace("../fruit/", "../fruit/")
    )


def _rel_asset(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _rewrite_template_asset_paths(root: ET.Element, output_dir: Path) -> None:
    replacements = {
        "../universal_robots_ur5e/": ASSET_ROOT / "universal_robots_ur5e",
        "../robotiq_2f85/": ASSET_ROOT / "robotiq_2f85",
        "../table/": ASSET_ROOT / "table",
        "../fruit/": ASSET_ROOT / "fruit",
    }
    for elem in root.iter():
        for attr in ("file", "meshdir", "texturedir"):
            value = elem.get(attr)
            if not value:
                continue
            for prefix, target_root in replacements.items():
                if value.startswith(prefix):
                    suffix = value[len(prefix):]
                    elem.set(attr, _rel_asset(target_root / suffix, output_dir))
                    break


def _spec_from_template(
    template: Path,
    camera_poses: dict[str, dict[str, list[float]]] | None,
    output_dir: Path,
) -> mujoco.MjSpec:
    if not template.exists():
        raise SceneGenerationError(f"Scene template not found: {template}")
    root = ET.fromstring(_template_text(template))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise SceneGenerationError(f"worldbody not found in template: {template}")
    _rewrite_template_asset_paths(root, output_dir)

    for camera in worldbody.findall("camera"):
        pose = (camera_poses or {}).get(camera.get("name", ""))
        if not pose:
            continue
        if "pos" in pose:
            pos = _ensure_vector3(pose["pos"], camera.get("name", "camera"))
            camera.set("pos", " ".join(f"{value:.9g}" for value in pos))
        if "quat" in pose:
            quat = _ensure_quat(pose["quat"], camera.get("name", "camera"))
            camera.set("quat", " ".join(f"{value:.9g}" for value in quat))
            camera.attrib.pop("xyaxes", None)

    xml_text = ET.tostring(root, encoding="unicode")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=str(output_dir), encoding="utf-8", delete=False) as handle:
        handle.write(xml_text)
        temp_path = Path(handle.name)
    try:
        return mujoco.MjSpec.from_file(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def _ensure_mesh(spec: mujoco.MjSpec, object_name: str, mesh_names: set[str]) -> str:
    if object_name.endswith("box"):
        return "box"
    mesh_name = object_name
    if mesh_name in mesh_names:
        return mesh_name
    mesh_path = FRUIT_STL_DIR / f"{mesh_name}.stl"
    if not mesh_path.exists():
        raise SceneGenerationError(f"Missing mesh for {object_name}: {mesh_path}")
    spec.add_mesh(file=str(mesh_path), name=mesh_name)
    mesh_names.add(mesh_name)
    return mesh_name


def _add_mesh_object(
    spec: mujoco.MjSpec,
    *,
    object_name: str,
    body_name: str,
    pos: list[float],
    material: str,
    object_quat: list[float] | None,
    mesh_quat: list[float] | None,
) -> None:
    body = spec.worldbody.add_body(
        name=body_name,
        quat=_compose_quats(object_quat, mesh_quat, body_name, [1.0, 0.0, 0.0, 0.0]),
        pos=pos,
    )
    body.add_joint(name=f"{body_name}_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(name=body_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=object_name, material=material)


def _add_box(spec: mujoco.MjSpec, *, body_name: str, pos: list[float], rgba: list[float]) -> None:
    body = spec.worldbody.add_body(name=body_name, quat=[1.0, 0.0, 0.0, 0.0], pos=pos)
    body.add_joint(name=f"{body_name}_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(name=body_name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.035, 0.035], rgba=rgba)


def _add_plate(spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body(name="plate", pos=[0.4, 0.4, 0.04])
    body.add_geom(
        name="plate_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.08, 0.008],
        rgba=[0.92, 0.88, 0.82, 1.0],
    )


def _box_rgba(name: str) -> list[float]:
    colors = {
        "red": [1.0, 0.0, 0.0, 1.0],
        "green": [0.0, 0.8, 0.1, 1.0],
        "blue": [0.0, 0.2, 1.0, 1.0],
        "yellow": [1.0, 0.9, 0.0, 1.0],
    }
    first = name.split("_")[0]
    return colors.get(first, [0.6, 0.6, 0.6, 1.0])


def generate_scene(
    result: dict[str, list[float]],
    *,
    scene_out: str | Path | None = None,
    template: str | Path | None = None,
    camera_poses: dict[str, dict[str, list[float]]] | None = None,
    object_quats: dict[str, list[float]] | None = None,
    mesh_quats: dict[str, list[float]] | None = None,
) -> Path:
    if not isinstance(result, dict) or not result:
        raise SceneGenerationError("result must be a non-empty dict of object_name -> [x, y, z].")

    template_path = Path(template) if template is not None else DEFAULT_TEMPLATE
    output_path = Path(scene_out) if scene_out is not None else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = _spec_from_template(template_path, camera_poses, output_path.parent)
    spec.option.timestep = 0.005
    spec.option.gravity = (0.0, 0.0, -9.81)
    _add_plate(spec)

    mesh_names: set[str] = set()
    object_counts: dict[str, int] = {}
    for object_name, raw_pos in result.items():
        normalized = str(object_name).strip().lower().replace(" ", "_")
        pos = _ensure_position(raw_pos, normalized)
        index = object_counts.get(normalized, 0)
        body_name = f"{normalized}{index}" if not normalized.endswith("box") else f"{normalized}_{index}"
        if normalized.endswith("box"):
            _add_box(spec, body_name=body_name, pos=pos, rgba=_box_rgba(normalized))
            object_counts[normalized] = index + 1
            continue
        _ensure_mesh(spec, normalized, mesh_names)
        material = "apple_mat" if normalized == APPLE else "banana_mat" if normalized == PEAR else ""
        _add_mesh_object(
            spec,
            object_name=normalized,
            body_name=body_name,
            pos=pos,
            material=material,
            object_quat=(object_quats or {}).get(normalized),
            mesh_quat=(mesh_quats or {}).get(normalized),
        )
        object_counts[normalized] = index + 1

    spec.compile()
    spec.to_file(str(output_path))
    return output_path.resolve()
