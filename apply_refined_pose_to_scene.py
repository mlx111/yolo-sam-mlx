#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from runtime_support_height import adjust_scene_support_heights


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime.xml"
DEFAULT_POSE_JSON = ROOT_DIR / "runtime_assets" / "left_view_refined_pose.json"
DEFAULT_OUT = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime_refined.xml"
BODY_BY_OBJECT = {"apple": "apple0", "pear": "pear0"}
MESH_BY_OBJECT = {
    "apple": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "apple.stl",
    "pear": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "pear.stl",
}
SUPPORT_CLEARANCE_M = 0.001


def _fmt(values: list[float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _quat_multiply_wxyz(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _normalized_quat_wxyz(q: list[float]) -> list[float]:
    import math

    values = [float(v) for v in q]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion.")
    return [v / norm for v in values]


def main() -> None:
    if not DEFAULT_SCENE.exists():
        raise FileNotFoundError(f"Missing scene XML: {DEFAULT_SCENE}")
    if not DEFAULT_POSE_JSON.exists():
        raise FileNotFoundError(f"Missing refined pose JSON: {DEFAULT_POSE_JSON}")

    payload = json.loads(DEFAULT_POSE_JSON.read_text(encoding="utf-8"))
    refined = payload.get("refined", {})
    calibration_path_raw = payload.get("calibration_path", "")
    calibration_path = Path(str(calibration_path_raw)).expanduser() if calibration_path_raw else None

    tree = ET.parse(DEFAULT_SCENE)
    root = tree.getroot()

    updated = {}
    for object_name, body_name in BODY_BY_OBJECT.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"Body not found in scene: {body_name}")
        pose = refined.get(object_name)
        if not isinstance(pose, dict):
            raise ValueError(f"Missing refined pose for {object_name}")
        pos = pose.get("pos")
        quat = pose.get("quat_wxyz")
        if pos is None or quat is None:
            raise ValueError(f"Incomplete refined pose for {object_name}")
        body.set("pos", _fmt(pos))
        body.set("quat", _fmt(quat))
        updated[object_name] = {
            "body": body_name,
            "pos": pos,
            "quat_wxyz": quat,
            "quat_source": "refined_pose_json",
        }

    tree.write(DEFAULT_OUT, encoding="utf-8", xml_declaration=True)
    support_adjustments = adjust_scene_support_heights(
        scene_path=DEFAULT_OUT,
        mesh_paths=MESH_BY_OBJECT,
        body_by_object=BODY_BY_OBJECT,
        support_z=0.0,
        clearance=SUPPORT_CLEARANCE_M,
    )
    print(
        json.dumps(
            {"scene_out": str(DEFAULT_OUT), "updated": updated, "support_height_adjustments": support_adjustments},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
