#!/usr/bin/env python3
"""
Right-side workspace crop with coordinate-consistent anchors.

This script regenerates right-side anchor clouds (apple/pear/roboticarm)
using the same "merge" coordinate transform as right global cloud, then
applies workspace crop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from pointcloud_v2 import PointCloudGenerator


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

RIGHT_GLOBAL_PATH = OUTPUT_DIR / "right1_background.npy"
RIGHT_WORKSPACE_NPY = OUTPUT_DIR / "right1_background_workspace_crop.npy"
RIGHT_WORKSPACE_PLY = OUTPUT_DIR / "right1_background_workspace_crop.ply"
RIGHT_SUMMARY_PATH = OUTPUT_DIR / "right1_background_workspace_crop_summary.json"
RIGHT_DEBUG_ANCHOR_DIR = OUTPUT_DIR / "right_workspace_anchors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop right global point cloud with consistent right-side anchors.")
    parser.add_argument("--margin-x", type=float, default=2.0)
    parser.add_argument("--margin-y", type=float, default=3.0)
    parser.add_argument("--margin-z", type=float, default=3.0)
    return parser.parse_args()


def _save_point_cloud(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def _load_npy_points(path: Path) -> np.ndarray:
    points = np.load(path)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError(f"Invalid point cloud: {path}")
    return points


def _right_generator_merge() -> PointCloudGenerator:
    # Keep right cloud transform consistent with merge-mode global cloud.
    return PointCloudGenerator(
        fx=1126.8856,
        fy=1126.4037,
        cx=954.9412,
        cy=536.3848,
        tcp_pose=[0, 0, 0, 0, 0, 0],
        camera_to_tcp_pose=[0, 0, 0, 0, 0, 0],
        visualize=False,
        save_point_cloud=False,
    )


def _generate_right_mask_points(object_name: str) -> np.ndarray:
    if object_name not in {"apple", "pear", "roboticarm"}:
        raise ValueError(f"Unsupported object: {object_name}")
    generator = _right_generator_merge()
    color_img = PointCloudGenerator.read_image_safely("inputs/cright001.png", is_depth=False)
    depth_img = PointCloudGenerator.read_image_safely("inputs/dright001.png", is_depth=True)
    response = generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask_path=f"inputs/right_mask_{object_name}.png",
        use_mask_auto=True,
        downsample_scale=1.0,
        objects=object_name,
        flag="right",
        type1="merge",
    )
    if response.get("state") != "success" or response.get("point_cloud") is None:
        raise RuntimeError(f"Failed to generate right {object_name} points: {response.get('info')}")
    points = np.asarray(response["point_cloud"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise RuntimeError(f"Invalid generated right {object_name} points.")
    return points


def _bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(points, axis=0), np.max(points, axis=0)


def crop_workspace_points(
    global_points: np.ndarray,
    arm_points: np.ndarray,
    apple_points: np.ndarray,
    pear_points: np.ndarray,
    margin_xyz: np.ndarray,
) -> tuple[np.ndarray, dict]:
    anchor_points = np.vstack([arm_points, apple_points, pear_points])
    bbox_min, bbox_max = _bounding_box(anchor_points)
    expanded_min = bbox_min - margin_xyz
    expanded_max = bbox_max + margin_xyz

    mask = np.all((global_points >= expanded_min) & (global_points <= expanded_max), axis=1)
    cropped_points = global_points[mask]

    summary = {
        "anchor_bbox_min": bbox_min.tolist(),
        "anchor_bbox_max": bbox_max.tolist(),
        "expanded_bbox_min": expanded_min.tolist(),
        "expanded_bbox_max": expanded_max.tolist(),
        "margin_xyz_m": margin_xyz.tolist(),
        "global_count": int(len(global_points)),
        "cropped_count": int(len(cropped_points)),
        "retention_ratio_percent": float(len(cropped_points) / len(global_points) * 100.0),
        "arm_count": int(len(arm_points)),
        "apple_count": int(len(apple_points)),
        "pear_count": int(len(pear_points)),
    }
    return cropped_points, summary


def main() -> None:
    args = parse_args()
    margin_xyz = np.array([float(args.margin_x), float(args.margin_y), float(args.margin_z)], dtype=float)

    global_points = _load_npy_points(RIGHT_GLOBAL_PATH)
    apple_points = _generate_right_mask_points("apple")
    pear_points = _generate_right_mask_points("pear")
    arm_points = _generate_right_mask_points("roboticarm")

    RIGHT_DEBUG_ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RIGHT_DEBUG_ANCHOR_DIR / "right_apple_anchor_merge.npy", apple_points)
    np.save(RIGHT_DEBUG_ANCHOR_DIR / "right_pear_anchor_merge.npy", pear_points)
    np.save(RIGHT_DEBUG_ANCHOR_DIR / "right_roboticarm_anchor_merge.npy", arm_points)

    cropped_points, summary = crop_workspace_points(
        global_points=global_points,
        arm_points=arm_points,
        apple_points=apple_points,
        pear_points=pear_points,
        margin_xyz=margin_xyz,
    )

    np.save(RIGHT_WORKSPACE_NPY, cropped_points)
    _save_point_cloud(cropped_points, RIGHT_WORKSPACE_PLY)
    RIGHT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 右侧一致坐标裁剪结果 ===")
    print(f"全局点数: {summary['global_count']}")
    print(f"裁剪后点数: {summary['cropped_count']}")
    print(f"保留比例: {summary['retention_ratio_percent']:.2f}%")
    print(f"机械臂点数: {summary['arm_count']}")
    print(f"苹果点数: {summary['apple_count']}")
    print(f"梨点数: {summary['pear_count']}")
    print(f"扩展包围盒最小值: {summary['expanded_bbox_min']}")
    print(f"扩展包围盒最大值: {summary['expanded_bbox_max']}")
    print(f"结果已保存至: {RIGHT_WORKSPACE_PLY}")


if __name__ == "__main__":
    main()
