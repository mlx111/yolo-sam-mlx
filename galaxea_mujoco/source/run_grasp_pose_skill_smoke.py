from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from skills.base.grasp_pose_skill import load_skill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R1Pro grasp/pregrasp pose matrices for an object.")
    parser.add_argument("--model", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--object-body", default="target_cube")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--grasp-mode", choices=["topdown", "side_x", "side_y", "current_rotation"], default="topdown")
    parser.add_argument("--pregrasp-distance", type=float, default=0.06)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    result = load_skill().compute_from_object(
        model,
        data,
        object_body=args.object_body,
        side=args.side,
        grasp_mode=args.grasp_mode,
        pregrasp_distance=args.pregrasp_distance,
        grasp_offset=np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64),
    )
    payload = {
        "schema_version": "r1pro_grasp_pose_skill_smoke_v1",
        "model": args.model,
        "result": asdict(result),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
