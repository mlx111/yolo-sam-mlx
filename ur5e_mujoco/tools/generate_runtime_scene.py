from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


UR5E_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UR5E_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ur5e_mujoco.runtime_scene import generate_scene

DEFAULT_SCENE_OUT = UR5E_ROOT / "scene" / "manual_scene.xml"
DEFAULT_REPORT_OUT = UR5E_ROOT / "scene" / "manual_report.json"


def _parse_object(raw: str) -> tuple[str, list[float]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Object must use name=x,y,z format, e.g. apple=0.62,0.18,0.78")
    name, coords_raw = raw.split("=", 1)
    coords = [float(item.strip()) for item in coords_raw.split(",") if item.strip()]
    if len(coords) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly x,y,z coordinates.")
    return name.strip().lower(), coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a UR5e MuJoCo runtime scene from object poses.")
    parser.add_argument(
        "--object",
        dest="objects",
        action="append",
        type=_parse_object,
        required=True,
        help="Object pose in meters, format name=x,y,z. Can be repeated.",
    )
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT))
    args = parser.parse_args()

    object_positions = dict(args.objects)
    scene_path = generate_scene(object_positions, scene_out=args.scene_out)
    report = {
        "status": "success",
        "scene_out": str(scene_path),
        "object_positions_m": object_positions,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
