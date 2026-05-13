from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import cv2
from matplotlib import colors
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parent

APPLE = "apple"
PEAR = "pear"


class SceneGenerationError(RuntimeError):
    """Raised when runtime scene generation cannot complete."""


def _scene_template() -> Path:
    return ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "scene2.xml"


def _scene_output() -> Path:
    return ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "s.xml"


def _fruit_mesh_dir() -> Path:
    return ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl"


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
    values = values / norm
    return values.tolist()


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
    if mesh_quat is None:
        combined = base_rotation
    else:
        combined = base_rotation @ _quat_to_matrix(mesh_quat, f"{name}_mesh")
    quat_xyzw = Rotation.from_matrix(combined).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _spec_from_template(
    template: Path,
    camera_poses: dict[str, dict[str, list[float]]] | None,
) -> mujoco.MjSpec:
    if not camera_poses:
        return mujoco.MjSpec.from_file(str(template))

    root = ET.fromstring(template.read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise SceneGenerationError(f"worldbody not found in template: {template}")

    for camera in worldbody.findall("camera"):
        pose = camera_poses.get(camera.get("name", ""))
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
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        dir=str(template.parent),
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(xml_text)
        temp_path = Path(handle.name)
    try:
        return mujoco.MjSpec.from_file(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def _add_apple(
    spec: mujoco.MjSpec,
    pos: list[float],
    index: int,
    quat: list[float] | None = None,
    mesh_quat: list[float] | None = None,
) -> None:
    body = spec.worldbody.add_body(
        name=f"apple{index}",
        quat=_compose_quats(quat, mesh_quat, f"apple{index}", [1.0, 0.0, 0.0, 0.0]),
        pos=pos,
    )
    body.add_joint(name=f"Apple_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"apple{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=APPLE,
        material="apple_mat",
    )


def _add_pear(
    spec: mujoco.MjSpec,
    pos: list[float],
    index: int,
    quat: list[float] | None = None,
    mesh_quat: list[float] | None = None,
) -> None:
    body = spec.worldbody.add_body(
        name=f"pear{index}",
        quat=_compose_quats(quat, mesh_quat, f"pear{index}", [1.0, 0.0, 0.0, 0.0]),
        pos=pos,
    )
    body.add_joint(name=f"pear_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"pear{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=PEAR,
        material="banana_mat",
    )


def _add_banana(spec: mujoco.MjSpec, pos: list[float], index: int) -> None:
    body = spec.worldbody.add_body(
        name=f"banana{index}",
        quat=[0.0, 1.0, 0.0, 0.0],
        pos=pos,
    )
    body.add_joint(name=f"banana_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"banana{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="banana",
        material="banana_mat",
    )


def _add_bowl(spec: mujoco.MjSpec, pos: list[float], index: int) -> None:
    body = spec.worldbody.add_body(
        name=f"bowl{index}",
        quat=[1.0, 0.0, 0.0, 0.0],
        pos=pos,
    )
    body.add_joint(name=f"bowl_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"bowl{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="bowl",
    )


def _add_camera(spec: mujoco.MjSpec, pos: list[float], index: int) -> None:
    body = spec.worldbody.add_body(
        name=f"camera{index}",
        quat=[1.0, 0.0, 0.0, 0.0],
        pos=pos,
    )
    body.add_joint(name=f"camera_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"camera{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="camera",
    )


def _add_cup(spec: mujoco.MjSpec, pos: list[float], index: int) -> None:
    body = spec.worldbody.add_body(name=f"cup{index}", pos=pos)
    body.add_joint(name=f"cup_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"cup{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cup",
    )


def _add_eyeglasses(spec: mujoco.MjSpec, pos: list[float], index: int) -> None:
    body = spec.worldbody.add_body(name=f"eyeglasses{index}", pos=pos)
    body.add_joint(name=f"eyeglasses_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"eyeglasses{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="eyeglasses",
    )


def _add_hammer(spec: mujoco.MjSpec, pos: list[float], index: int) -> None:
    body = spec.worldbody.add_body(name=f"hammer{index}", pos=pos)
    body.add_joint(name=f"hammer_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"hammer{index}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="hammer",
    )


def _add_box(spec: mujoco.MjSpec, pos: list[float], index: int, rgba: tuple[float, float, float, float]) -> None:
    body = spec.worldbody.add_body(
        name=f"box_{index}",
        quat=[1.0, 0.0, 0.0, 0.0],
        pos=pos,
    )
    body.add_joint(name=f"box_joint{index}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    body.add_geom(
        name=f"box_{index}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.035, 0.035, 0.035],
        rgba=rgba,
    )


OBJECT_ADDERS: dict[str, Callable[[mujoco.MjSpec, list[float], int], None]] = {
    APPLE: _add_apple,
    PEAR: _add_pear,
    "banana": _add_banana,
    "bowl": _add_bowl,
    "camera": _add_camera,
    "cup": _add_cup,
    "eyeglasses": _add_eyeglasses,
    "hammer": _add_hammer,
}


def _ensure_mesh(spec: mujoco.MjSpec, object_name: str, mesh_names: set[str]) -> str:
    if object_name.endswith("box"):
        return "box"
    mesh_name = "eyeglasses" if object_name == "eyegalsses" else object_name
    if mesh_name in mesh_names:
        return mesh_name

    mesh_path = _fruit_mesh_dir() / f"{mesh_name}.stl"
    if not mesh_path.exists():
        raise SceneGenerationError(f"Missing mesh for {object_name}: {mesh_path}")
    spec.add_mesh(file=str(mesh_path), name=mesh_name)
    mesh_names.add(mesh_name)
    return mesh_name


def _render_depth_frame(model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int) -> None:
    color_renderer = mujoco.renderer.Renderer(model, width, height)
    depth_renderer = mujoco.renderer.Renderer(model, width, height)
    color_renderer.update_scene(data, 0)
    depth_renderer.update_scene(data, 0)
    depth_renderer.enable_depth_rendering()
    color_img = color_renderer.render()
    _ = depth_renderer.render()
    cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)


def generate_scene(
    result: dict[str, list[float]],
    visual: bool = False,
    scene_out: str | os.PathLike[str] | None = None,
    camera_poses: dict[str, dict[str, list[float]]] | None = None,
    object_quats: dict[str, list[float]] | None = None,
    mesh_quats: dict[str, list[float]] | None = None,
) -> Path:
    """
    Build a MuJoCo scene directly from zhenghe.py style `result`.

    `result` format:
        {
            "apple": [x, y, z],
            "pear": [x, y, z],
            ...
        }
    """
    if not isinstance(result, dict) or not result:
        raise SceneGenerationError("result must be a non-empty dict of object_name -> [x, y, z].")

    template = _scene_template()
    if not template.exists():
        raise SceneGenerationError(f"Scene template not found: {template}")

    spec = _spec_from_template(template, camera_poses)
    spec.option.timestep = 0.005
    spec.option.gravity = (0.0, 0.0, -9.81)

    mesh_names: set[str] = set()
    object_counts: dict[str, int] = {}

    for object_name, raw_pos in result.items():
        normalized_name = "eyeglasses" if object_name == "eyegalsses" else object_name
        pos = _ensure_position(raw_pos, object_name)
        count = object_counts.get(normalized_name, 0)

        if normalized_name.endswith("box"):
            words = normalized_name.split("_")
            rgba_name = words[0] if words else "gray"
            rgba = colors.to_rgba(rgba_name)
            _add_box(spec, pos, count, rgba)
            object_counts[normalized_name] = count + 1
            continue

        _ensure_mesh(spec, normalized_name, mesh_names)
        add_fn = OBJECT_ADDERS.get(normalized_name)
        if add_fn is None:
            raise SceneGenerationError(f"Unsupported object type: {object_name}")
        if normalized_name in {APPLE, PEAR} and (
            (object_quats and normalized_name in object_quats)
            or (mesh_quats and normalized_name in mesh_quats)
        ):
            quat = None
            mesh_quat = None
            if object_quats and normalized_name in object_quats:
                quat = _ensure_quat(object_quats[normalized_name], normalized_name)
            if mesh_quats and normalized_name in mesh_quats:
                mesh_quat = _ensure_quat(mesh_quats[normalized_name], f"{normalized_name}_mesh")
            if normalized_name == APPLE:
                _add_apple(spec, pos, count, quat=quat, mesh_quat=mesh_quat)
            else:
                _add_pear(spec, pos, count, quat=quat, mesh_quat=mesh_quat)
            object_counts[normalized_name] = count + 1
            continue
        add_fn(spec, pos, count)
        object_counts[normalized_name] = count + 1

    model = spec.compile()
    data = mujoco.MjData(model)

    if visual:
        import glfw
        import time
        from mujoco import viewer as mj_viewer

        if not glfw.init():
            raise SceneGenerationError("GLFW initialization failed.")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(1200, 900, "mujoco", None, None)
        if window is None:
            glfw.terminate()
            raise SceneGenerationError("GLFW window creation failed.")
        glfw.make_context_current(window)
        viewer = mj_viewer.launch_passive(model, data)
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                _render_depth_frame(model, data, 640, 480)
                viewer.sync()
                time.sleep(0.002)
        finally:
            viewer.close()
            glfw.destroy_window(window)
            glfw.terminate()

    out_path = Path(scene_out).resolve() if scene_out else _scene_output().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(spec.to_xml(), encoding="utf-8")
    return out_path


generate_scene_from_result = generate_scene


def gen_prompt(objects: dict[str, list[float]]) -> str:
    """
    Compatibility helper for old zhenghe.py style imports.
    """
    parts = []
    for key, value in objects.items():
        parts.append(f"{key},{value}")
    return ",".join(parts)


if __name__ == "__main__":
    sample = {
        "apple": [0.514164, 0.246903, 0.0265855],
        "pear": [0.410833, 0.0512228, 0.0196004],
    }
    output = generate_scene(sample, visual=False)
    print(output)
