#!/usr/bin/env python3
"""
围绕机械臂与目标物体的工作区裁剪左侧全局点云。
"""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from pointcloud_v2 import CAMERA_EULER_DEG, PointCloudGenerator
from calibrate_runtime_pose_from_clouds import _generator_for


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class WorkspacePaths:
    global_path: Path
    apple_path: Path
    pear_path: Path
    arm_cache_path: Path
    arm_cache_ply_path: Path
    workspace_ply_path: Path
    workspace_npy_path: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop workspace points from left/right global cloud.")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    return parser.parse_args()


def resolve_paths(side: str) -> WorkspacePaths:
    side_name = str(side).strip().lower()
    if side_name not in {"left", "right"}:
        raise ValueError(f"Unsupported side: {side}")
    return WorkspacePaths(
        global_path=OUTPUT_DIR / f"{side_name}1_background.npy",
        apple_path=OUTPUT_DIR / f"{side_name}_apple.npy",
        pear_path=OUTPUT_DIR / f"{side_name}_pear.npy",
        arm_cache_path=OUTPUT_DIR / f"{side_name}_roboticarm_workspace.npy",
        arm_cache_ply_path=OUTPUT_DIR / f"{side_name}_roboticarm_workspace.ply",
        workspace_ply_path=OUTPUT_DIR / f"{side_name}1_background_workspace_crop.ply",
        workspace_npy_path=OUTPUT_DIR / f"{side_name}1_background_workspace_crop.npy",
        summary_path=OUTPUT_DIR / f"{side_name}1_background_workspace_crop_summary.json",
    )


def _save_point_cloud(points: np.ndarray, path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def _load_npy_points(path: Path) -> np.ndarray:
    points = np.load(path)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError(f"Invalid point cloud: {path}")
    return points


def _generate_roboticarm_points_world(side: str) -> np.ndarray:
    side_name = str(side).strip().lower()
    if side_name not in {"left", "right"}:
        raise ValueError(f"Unsupported side: {side}")
    rx, ry, rz = CAMERA_EULER_DEG[side_name]
    generator = _generator_for(side_name, rx, ry, rz)
    color_img = PointCloudGenerator.read_image_safely(f"inputs/c{side_name}001.png", is_depth=False)
    depth_img = PointCloudGenerator.read_image_safely(f"inputs/d{side_name}001.png", is_depth=True)
    response = generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask_path=f"inputs/{side_name}_mask_roboticarm.png",
        use_mask_auto=True,
        downsample_scale=1.0,
        objects="roboticarm",
        flag=side_name,
        type1="normal",
    )
    if response.get("state") != "success" or response.get("point_cloud") is None:
        raise ValueError(f"Unable to generate {side_name} roboticarm point cloud.")
    points = np.asarray(response["point_cloud"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError(f"Invalid {side_name} roboticarm point cloud.")
    return points


def _ensure_arm_points(paths: WorkspacePaths, side: str) -> np.ndarray:
    if paths.arm_cache_path.exists():
        return _load_npy_points(paths.arm_cache_path)
    points = _generate_roboticarm_points_world(side)
    np.save(paths.arm_cache_path, points)
    _save_point_cloud(points, paths.arm_cache_ply_path)
    return points


def _generator_for_merge(side: str) -> PointCloudGenerator:
    side_name = str(side).strip().lower()
    if side_name == "left":
        return PointCloudGenerator(
            fx=1129.8136,
            fy=1128.6075,
            cx=961.0022,
            cy=546.8298,
            tcp_pose=[0, 0, 0, 0, 0, 0],
            camera_to_tcp_pose=[0, 0, 0, 0, 0, 0],
            visualize=False,
            save_point_cloud=False,
        )
    if side_name == "right":
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
    raise ValueError(f"Unsupported side: {side}")


def _generate_mask_points_merge(side: str, object_name: str) -> np.ndarray:
    side_name = str(side).strip().lower()
    if object_name not in {"apple", "pear", "roboticarm"}:
        raise ValueError(f"Unsupported object: {object_name}")
    generator = _generator_for_merge(side_name)
    color_img = PointCloudGenerator.read_image_safely(f"inputs/c{side_name}001.png", is_depth=False)
    depth_img = PointCloudGenerator.read_image_safely(f"inputs/d{side_name}001.png", is_depth=True)
    response = generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask_path=f"inputs/{side_name}_mask_{object_name}.png",
        use_mask_auto=True,
        downsample_scale=1.0,
        objects=object_name,
        flag=side_name,
        type1="merge",
    )
    if response.get("state") != "success" or response.get("point_cloud") is None:
        raise ValueError(f"Unable to generate {side_name} {object_name} point cloud in merge mode.")
    points = np.asarray(response["point_cloud"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError(f"Invalid {side_name} {object_name} merge point cloud.")
    return points


def load_anchor_points(paths: WorkspacePaths, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side_name = str(side).strip().lower()
    if side_name == "right":
        # Right global cloud is generated in merge coordinates in this project.
        # Rebuild anchors in the same coordinates to avoid bbox mismatch.
        arm_points = _generate_mask_points_merge(side_name, "roboticarm")
        apple_points = _generate_mask_points_merge(side_name, "apple")
        pear_points = _generate_mask_points_merge(side_name, "pear")
        return arm_points, apple_points, pear_points

    apple_points = _load_npy_points(paths.apple_path)
    pear_points = _load_npy_points(paths.pear_path)
    arm_points = _ensure_arm_points(paths, side_name)
    return arm_points, apple_points, pear_points


def _bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(points, axis=0), np.max(points, axis=0)


def crop_workspace_points(
    global_points: np.ndarray,
    arm_points: np.ndarray,
    apple_points: np.ndarray,
    pear_points: np.ndarray,
    margin_min_xyz: np.ndarray | None = None,
    margin_max_xyz: np.ndarray | None = None,
    margin_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if margin_xyz is not None:
        margin_min = np.asarray(margin_xyz, dtype=float)
        margin_max = np.asarray(margin_xyz, dtype=float)
    else:
        if margin_min_xyz is None or margin_max_xyz is None:
            raise ValueError("Provide margin_xyz or both margin_min_xyz/margin_max_xyz.")
        margin_min = np.asarray(margin_min_xyz, dtype=float)
        margin_max = np.asarray(margin_max_xyz, dtype=float)

    if margin_min.shape != (3,) or margin_max.shape != (3,):
        raise ValueError("Margins must be 3D vectors [x, y, z].")

    anchor_points = np.vstack([arm_points, apple_points, pear_points])
    bbox_min, bbox_max = _bounding_box(anchor_points)
    expanded_min = bbox_min - margin_min
    expanded_max = bbox_max + margin_max

    mask = np.all((global_points >= expanded_min) & (global_points <= expanded_max), axis=1)
    cropped_points = global_points[mask]

    summary = {
        "anchor_bbox_min": bbox_min.tolist(),
        "anchor_bbox_max": bbox_max.tolist(),
        "expanded_bbox_min": expanded_min.tolist(),
        "expanded_bbox_max": expanded_max.tolist(),
        "margin_min_xyz_m": margin_min.tolist(),
        "margin_max_xyz_m": margin_max.tolist(),
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
    side = str(args.side).strip().lower()
    paths = resolve_paths(side)

    global_points = _load_npy_points(paths.global_path)
    arm_points, apple_points, pear_points = load_anchor_points(paths, side)

    # Edit these six values to control how much to keep on each axis end:
    # expanded_min = anchor_min - margin_min_xyz
    # expanded_max = anchor_max + margin_max_xyz
    margin_min_xyz = np.array([0.1, 0.5, 0.5], dtype=float)  # [x_min, y_min, z_min]
    margin_max_xyz = np.array([0.9, 0.9, 1.4], dtype=float)  # [x_max, y_max, z_max]
    cropped_points, summary = crop_workspace_points(
        global_points=global_points,
        arm_points=arm_points,
        apple_points=apple_points,
        pear_points=pear_points,
        margin_min_xyz=margin_min_xyz,
        margin_max_xyz=margin_max_xyz,
    )

    np.save(paths.workspace_npy_path, cropped_points)
    _save_point_cloud(cropped_points, paths.workspace_ply_path)
    paths.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 工作区裁剪结果 ===")
    print(f"全局点数: {summary['global_count']}")
    print(f"裁剪后点数: {summary['cropped_count']}")
    print(f"保留比例: {summary['retention_ratio_percent']:.2f}%")
    print(f"机械臂点数: {summary['arm_count']}")
    print(f"苹果点数: {summary['apple_count']}")
    print(f"梨点数: {summary['pear_count']}")
    print(f"下限扩展(m): {summary['margin_min_xyz_m']}")
    print(f"上限扩展(m): {summary['margin_max_xyz_m']}")
    print(f"扩展包围盒最小值: {summary['expanded_bbox_min']}")
    print(f"扩展包围盒最大值: {summary['expanded_bbox_max']}")
    print(f"处理侧别: {side}")
    print(f"结果已保存至: {paths.workspace_ply_path}")


if __name__ == "__main__":
    main()
