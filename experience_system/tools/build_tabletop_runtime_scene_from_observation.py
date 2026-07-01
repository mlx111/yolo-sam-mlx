"""Convert a field tabletop observation into a runtime sandbox scene JSON.

This tool is intentionally narrow for the Galaxea tabletop experiments: the
table is fixed by defaults, while objects/place zones come from the task-start
observation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import RuntimeSandboxScene


FIXED_TABLE_POSE = [0.2, 0.0, 0.737]
FIXED_TABLE_SIZE = [0.35, 0.35, 0.025]
DEFAULT_ROBOT_MODEL_INCLUDE = "model/model.xml"
DEFAULT_TIMESTEP = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build runtime_sandbox_scene_v1 from field_runtime_scene_observation_v1.")
    parser.add_argument("--observation", type=Path, required=True, help="field_runtime_scene_observation_v1 JSON")
    parser.add_argument("--save-runtime-scene", type=Path, required=True, help="output runtime_sandbox_scene_v1 JSON")
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--robot-model-include", default="", help="override runtime scene robot include")
    parser.add_argument("--target-object", default="", help="override target object body name")
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_item(item: dict[str, Any], *, fallback_name: str) -> dict[str, Any]:
    out = {
        "name": str(item.get("name") or fallback_name),
        "pose": item.get("pose") or item.get("position") or [0.0, 0.0, 0.8],
        "size": item.get("size") or [0.03, 0.03, 0.03],
        "geom_type": item.get("geom_type") or item.get("type") or "box",
        "mass": item.get("mass", 0.05),
        "freejoint": item.get("freejoint", True),
    }
    for key in ("rgba", "friction", "class", "confidence", "source"):
        if key in item:
            out[key] = item[key]
    return out


def _runtime_scene_from_observation(payload: dict[str, Any], *, target_object: str = "", robot_model_include: str = "", timestep: float = DEFAULT_TIMESTEP) -> dict[str, Any]:
    runtime = _as_dict(payload.get("runtime_scene"))
    objects = [
        _clean_item(item, fallback_name=f"object_{index}")
        for index, item in enumerate(_as_list(payload.get("objects")))
        if isinstance(item, dict)
    ]
    obstacles = [
        _clean_item(item, fallback_name=f"obstacle_{index}")
        for index, item in enumerate(_as_list(payload.get("obstacles")))
        if isinstance(item, dict)
    ]
    place_zones = [
        {
            "name": str(item.get("name") or f"place_zone_{index}"),
            "pose": item.get("pose") or item.get("position") or [0.0, 0.25, 0.805],
            "size": item.get("size") or [0.08, 0.06, 0.008],
            **({"rgba": item["rgba"]} if "rgba" in item else {}),
            **({"class": item["class"]} if "class" in item else {}),
            **({"confidence": item["confidence"]} if "confidence" in item else {}),
            **({"source": item["source"]} if "source" in item else {}),
        }
        for index, item in enumerate(_as_list(payload.get("place_zones")))
        if isinstance(item, dict)
    ]
    selected_target = target_object or str(runtime.get("target_object") or (objects[0]["name"] if objects else ""))
    selected_include = robot_model_include or str(runtime.get("robot_model_include") or DEFAULT_ROBOT_MODEL_INCLUDE)
    selected_timestep = float(timestep if timestep is not None else runtime.get("timestep", DEFAULT_TIMESTEP))
    scene = {
        "schema_version": "runtime_sandbox_scene_v1",
        "scene_id": str(payload.get("scene_id") or "runtime_tabletop_scene"),
        "robot_model_include": selected_include,
        "timestep": selected_timestep,
        "table_pose": list(FIXED_TABLE_POSE),
        "table_size": list(FIXED_TABLE_SIZE),
        "target_object": selected_target,
        "objects": objects,
        "obstacles": obstacles,
        "place_zones": place_zones,
        "metadata": {
            "source_schema_version": payload.get("schema_version"),
            "source_scene_id": payload.get("scene_id"),
            "timestamp": payload.get("timestamp"),
            "coordinate_frame": _as_dict(payload.get("coordinate_frame")),
            "robot_state": _as_dict(payload.get("robot_state")),
            "sensor_refs": _as_dict(payload.get("sensor_refs")),
            "calibration": _as_dict(payload.get("calibration")),
            "observation_metadata": _as_dict(payload.get("metadata")),
            "table_policy": "fixed_table_from_galaxea_tabletop_defaults",
            "task_start_runtime_scene": True,
        },
    }
    RuntimeSandboxScene.from_dict(scene)
    return scene


def main() -> None:
    args = parse_args()
    observation = json.loads(args.observation.read_text(encoding="utf-8"))
    if not isinstance(observation, dict):
        raise ValueError("--observation must contain a JSON object")
    scene = _runtime_scene_from_observation(
        observation,
        target_object=args.target_object,
        robot_model_include=args.robot_model_include,
        timestep=args.timestep,
    )
    _write_json(args.save_runtime_scene, scene)
    report = {
        "schema_version": "tabletop_runtime_scene_conversion_report_v1",
        "observation": str(args.observation),
        "runtime_scene": str(args.save_runtime_scene),
        "scene_id": scene["scene_id"],
        "target_object": scene["target_object"],
        "object_count": len(scene["objects"]),
        "obstacle_count": len(scene["obstacles"]),
        "place_zone_count": len(scene["place_zones"]),
        "table_policy": scene["metadata"]["table_policy"],
    }
    if args.save_report is not None:
        _write_json(args.save_report, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
