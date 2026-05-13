from __future__ import annotations

import argparse
import sys

import mujoco

from new_runtime.apple_pear_scene import SceneBuildError, artifacts_to_json, build_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone MuJoCo scene containing the robot, table, apple, and pear."
    )
    parser.add_argument("--mesh-source", choices=["fixed", "buquan"], default="fixed")
    parser.add_argument(
        "--camera",
        choices=["left", "right"],
        default="left",
        help="Preferred camera for buquan mesh generation. Positions still default to pointcloud_v2.pos(...).",
    )
    parser.add_argument("--scene-out", default="", help="Output XML path.")
    parser.add_argument("--apple-pos", default="", help="Explicit apple position: x,y,z")
    parser.add_argument("--pear-pos", default="", help="Explicit pear position: x,y,z")
    parser.add_argument("--validate-load", action="store_true", help="Load the generated XML with MuJoCo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifacts = build_scene(
            mesh_source=args.mesh_source,
            camera=args.camera,
            scene_out=args.scene_out or None,
            apple_pos=args.apple_pos or None,
            pear_pos=args.pear_pos or None,
        )
        if args.validate_load:
            mujoco.MjModel.from_xml_path(str(artifacts.scene_xml))
    except SceneBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(artifacts_to_json(artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
