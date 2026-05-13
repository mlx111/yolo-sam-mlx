#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from new_runtime.apple_pear_scene import (
    SceneBuildError,
    load_mesh_quats_from_selected_reports,
    resolve_meshes,
    selected_runner_reports,
)


def _parse_objects(raw: str) -> tuple[str, ...]:
    values = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise SceneBuildError("No objects specified. Use --objects apple,pear")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build runtime STL files (point cloud -> completion -> STL) and install into manipulator_grasp/assets/fruit/stl."
    )
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--objects", default="apple,pear", help="Comma-separated objects, e.g. apple,pear")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild instead of reusing cached buquan outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = _parse_objects(args.objects)

    mesh_files, report_files = resolve_meshes(
        mesh_source="buquan",
        camera=args.camera,
        objects=objects,
        force_rebuild=bool(args.force),
    )
    stable_reports = selected_runner_reports(objects)
    mesh_quats = load_mesh_quats_from_selected_reports(objects=objects, require_all=True)

    payload = {
        "camera": args.camera,
        "objects": list(objects),
        "force_rebuild": bool(args.force),
        "installed_stl": {name: str(path) for name, path in mesh_files.items()},
        "selected_runner_reports": {name: str(path) for name, path in stable_reports.items()},
        "resolved_reports": {name: str(path) for name, path in report_files.items()},
        "mesh_quats": mesh_quats,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SceneBuildError as exc:
        raise SystemExit(f"[ERROR] {exc}")
