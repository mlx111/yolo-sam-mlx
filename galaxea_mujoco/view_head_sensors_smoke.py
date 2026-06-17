import argparse

import mujoco
import numpy as np

from skills.base.base_lidar_skill import load_skill as load_base_lidar
from skills.base.head_camera_skill import load_skill as load_head_camera


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test external head RGB-D camera and chassis lidar skills.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--ray-count", type=int, default=21)
    return parser.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = load_head_camera(width=args.width, height=args.height)
    frame = camera.capture(model, data, include_depth=True)
    print(
        "external_head_rgbd_camera:",
        {
            "rgb_shape": list(frame.rgb.shape),
            "depth_shape": list(frame.depth.shape) if frame.depth is not None else None,
            "rgb_mean": round(float(np.mean(frame.rgb)), 6),
            "depth_min": round(float(np.nanmin(frame.depth)), 6) if frame.depth is not None else None,
            "depth_max": round(float(np.nanmax(frame.depth)), 6) if frame.depth is not None else None,
            "camera_position": np.round(frame.camera_position, 6).tolist(),
        },
    )

    lidar = load_base_lidar()
    scan = lidar.scan(model, data, ray_count=args.ray_count)
    finite_hits = int(np.sum(scan.hit_geom_ids >= 0))
    print(
        "base_lidar:",
        {
            "ray_count": int(scan.ranges.shape[0]),
            "finite_hits": finite_hits,
            "range_min": round(float(np.min(scan.ranges)), 6),
            "range_max": round(float(np.max(scan.ranges)), 6),
            "origin": np.round(scan.origin, 6).tolist(),
        },
    )

    if frame.rgb.shape != (args.height, args.width, 3):
        raise SystemExit("unexpected rgb shape")
    if frame.depth is None or frame.depth.shape != (args.height, args.width):
        raise SystemExit("unexpected depth shape")
    if scan.ranges.shape != (args.ray_count,):
        raise SystemExit("unexpected lidar shape")


if __name__ == "__main__":
    main()
