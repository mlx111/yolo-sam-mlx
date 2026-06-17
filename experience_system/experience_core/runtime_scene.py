"""Runtime sandbox scene description and MuJoCo XML generation."""

from __future__ import annotations

import html
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _vec(value: Any, *, default: list[float], n: int = 3) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return list(default)
    out: list[float] = []
    for item in list(value)[:n]:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return list(default)
    while len(out) < n:
        out.append(float(default[len(out)] if len(default) > len(out) else 0.0))
    return out


def _xml_vec(values: list[float]) -> str:
    return " ".join(f"{float(item):.6g}" for item in values)


def _safe_name(value: str, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).strip())
    return text or fallback


@dataclass
class RuntimeObjectSpec:
    name: str
    pose: list[float]
    size: list[float]
    geom_type: str = "box"
    mass: float = 0.05
    rgba: list[float] = field(default_factory=lambda: [0.9, 0.3, 0.15, 1.0])
    friction: list[float] = field(default_factory=lambda: [1.0, 0.02, 0.002])
    freejoint: bool = True
    role: str = "object"

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, index: int, role: str) -> "RuntimeObjectSpec":
        return cls(
            name=_safe_name(str(payload.get("name") or f"{role}_{index}"), f"{role}_{index}"),
            pose=_vec(payload.get("pose") or payload.get("position"), default=[0.0, 0.0, 0.8]),
            size=_vec(payload.get("size"), default=[0.03, 0.03, 0.03]),
            geom_type=str(payload.get("geom_type") or payload.get("type") or "box"),
            mass=float(payload.get("mass") or 0.05),
            rgba=_vec(payload.get("rgba"), default=[0.9, 0.3, 0.15, 1.0], n=4),
            friction=_vec(payload.get("friction"), default=[1.0, 0.02, 0.002]),
            freejoint=bool(payload.get("freejoint", True)),
            role=role,
        )


@dataclass
class RuntimePlaceZoneSpec:
    name: str
    pose: list[float]
    size: list[float]
    rgba: list[float] = field(default_factory=lambda: [0.1, 0.7, 0.25, 0.45])

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, index: int) -> "RuntimePlaceZoneSpec":
        return cls(
            name=_safe_name(str(payload.get("name") or f"place_zone_{index}"), f"place_zone_{index}"),
            pose=_vec(payload.get("pose") or payload.get("position"), default=[0.0, 0.25, 0.805]),
            size=_vec(payload.get("size"), default=[0.08, 0.06, 0.008]),
            rgba=_vec(payload.get("rgba"), default=[0.1, 0.7, 0.25, 0.45], n=4),
        )


@dataclass
class RuntimeSandboxScene:
    schema_version: str = "runtime_sandbox_scene_v1"
    scene_id: str = "runtime_scene"
    robot_model_include: str = "model.xml"
    timestep: float = 0.005
    table_pose: list[float] = field(default_factory=lambda: [0.2, 0.0, 0.737])
    table_size: list[float] = field(default_factory=lambda: [0.35, 0.35, 0.025])
    target_object: str = ""
    objects: list[RuntimeObjectSpec] = field(default_factory=list)
    obstacles: list[RuntimeObjectSpec] = field(default_factory=list)
    place_zones: list[RuntimePlaceZoneSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeSandboxScene":
        objects = [
            RuntimeObjectSpec.from_dict(item, index=index, role="object")
            for index, item in enumerate(payload.get("objects") or [])
            if isinstance(item, dict)
        ]
        obstacles = [
            RuntimeObjectSpec.from_dict(item, index=index, role="obstacle")
            for index, item in enumerate(payload.get("obstacles") or [])
            if isinstance(item, dict)
        ]
        place_zones = [
            RuntimePlaceZoneSpec.from_dict(item, index=index)
            for index, item in enumerate(payload.get("place_zones") or [])
            if isinstance(item, dict)
        ]
        return cls(
            scene_id=_safe_name(str(payload.get("scene_id") or "runtime_scene"), "runtime_scene"),
            robot_model_include=str(payload.get("robot_model_include") or "model.xml"),
            timestep=float(payload.get("timestep") or 0.005),
            table_pose=_vec(payload.get("table_pose"), default=[0.2, 0.0, 0.737]),
            table_size=_vec(payload.get("table_size"), default=[0.35, 0.35, 0.025]),
            target_object=str(payload.get("target_object") or (objects[0].name if objects else "")),
            objects=objects,
            obstacles=obstacles,
            place_zones=place_zones,
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _object_xml(item: RuntimeObjectSpec) -> str:
    name = html.escape(item.name)
    body = [f'    <body name="{name}" pos="{_xml_vec(item.pose)}">']
    if item.freejoint:
        body.append(f'      <freejoint name="{name}_freejoint"/>')
    body.append(
        f'      <geom name="{name}_geom" type="{html.escape(item.geom_type)}" '
        f'size="{_xml_vec(item.size)}" mass="{item.mass:.6g}" '
        f'rgba="{_xml_vec(item.rgba)}" condim="6" friction="{_xml_vec(item.friction)}" '
        'priority="1" solref="0.004 1" solimp="0.95 0.99 0.001"/>'
    )
    body.append(f'      <site name="{name}_site" pos="0 0 0" size="0.012" rgba="1 1 0.1 1"/>')
    body.append("    </body>")
    return "\n".join(body)


def _place_zone_xml(item: RuntimePlaceZoneSpec) -> str:
    name = html.escape(item.name)
    geom_pos = [item.pose[0], item.pose[1], item.pose[2] - 0.043]
    return "\n".join([
        f'    <geom name="{name}_geom" type="box" pos="{_xml_vec(geom_pos)}" size="{_xml_vec(item.size)}" '
        f'rgba="{_xml_vec(item.rgba)}" contype="0" conaffinity="0"/>',
        f'    <site name="{name}_site" pos="{_xml_vec(item.pose)}" size="0.015" rgba="0.1 1 0.2 1"/>',
    ])


def render_runtime_scene_xml(scene: RuntimeSandboxScene) -> str:
    lines = [
        f'<mujoco model="{html.escape(scene.scene_id)}">',
        f'  <include file="{html.escape(scene.robot_model_include)}"/>',
        f'  <option timestep="{scene.timestep:.6g}" impratio="10" integrator="implicitfast" cone="elliptic" solver="PGS"/>',
        "",
        "  <worldbody>",
        '    <light name="runtime_scene_key_light" pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>',
        f'    <geom name="runtime_table_geom" type="box" pos="{_xml_vec(scene.table_pose)}" size="{_xml_vec(scene.table_size)}" '
        'rgba="0.45 0.38 0.30 1" condim="3" friction="1.0 0.02 0.002" priority="1" solref="0.004 1" solimp="0.95 0.99 0.001"/>',
    ]
    for item in scene.objects:
        lines.append(_object_xml(item))
    for item in scene.obstacles:
        lines.append(_object_xml(item))
    for item in scene.place_zones:
        lines.append(_place_zone_xml(item))
    lines.extend(["  </worldbody>", "</mujoco>", ""])
    return "\n".join(lines)


def write_runtime_scene(scene: RuntimeSandboxScene, output: str | Path, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene_to_write = scene
    include = Path(scene.robot_model_include)
    if not include.is_absolute():
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        include_abs = (root / include).resolve()
        if include_abs.exists():
            scene_to_write = RuntimeSandboxScene.from_dict({
                **scene.to_dict(),
                "robot_model_include": os.path.relpath(include_abs, output.parent.resolve()),
            })
    output.write_text(render_runtime_scene_xml(scene_to_write), encoding="utf-8")
    return {
        "scene_id": scene_to_write.scene_id,
        "output": str(output),
        "robot_model_include": scene_to_write.robot_model_include,
        "target_object": scene_to_write.target_object,
        "object_count": len(scene_to_write.objects),
        "obstacle_count": len(scene_to_write.obstacles),
        "place_zone_count": len(scene_to_write.place_zones),
    }
