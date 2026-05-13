#!/usr/bin/env python3
"""
对 left1_background.ply 运行三组去背景参数，便于直接比较结果。
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from simple_background_removal import SimpleBackgroundRemoval


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
INPUT_PATH = OUTPUT_DIR / "left1_background.ply"
SUMMARY_PATH = OUTPUT_DIR / "left1_background_no_background_compare_summary.json"

PARAMETER_SETS = {
    "conservative": {
        "plane_threshold": 0.02,
        "plane_min_points": 3500,
        "distance_percentile": 90,
        "z_min_percentile": 5,
        "z_max_percentile": 95,
    },
    "balanced": {
        "plane_threshold": 0.015,
        "plane_min_points": 2000,
        "distance_percentile": 85,
        "z_min_percentile": 5,
        "z_max_percentile": 95,
    },
    "aggressive": {
        "plane_threshold": 0.012,
        "plane_min_points": 1200,
        "distance_percentile": 80,
        "z_min_percentile": 8,
        "z_max_percentile": 92,
    },
}


def _pcd_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _remove_large_planes_with_stats(
    remover: SimpleBackgroundRemoval,
    pcd: o3d.geometry.PointCloud,
    threshold: float,
    min_points: int,
    max_planes: int = 5,
):
    remaining_pcd = pcd
    total_removed = 0
    planes = []

    print("\n=== 去除大平面 ===")
    print(f"平面检测阈值: {threshold}")
    print(f"最小平面点数: {min_points}")

    for index in range(max_planes):
        if len(remaining_pcd.points) < min_points:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inliers) < min_points:
            break

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        a, b, c, d = plane_model
        plane_info = {
            "index": index + 1,
            "model": [float(a), float(b), float(c), float(d)],
            "points": int(len(inliers)),
        }
        planes.append(plane_info)
        total_removed += len(inliers)

        print(
            f"平面 {index + 1}: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0"
        )
        print(f"平面点数: {len(inliers)}")

    print(f"总共去除平面点数: {total_removed}")
    print(f"剩余点数: {len(remaining_pcd.points)}")

    return remaining_pcd, {
        "planes": planes,
        "total_removed": int(total_removed),
        "remaining_after_planes": int(len(remaining_pcd.points)),
    }


def _filter_by_distance_percentile(
    remover: SimpleBackgroundRemoval,
    pcd: o3d.geometry.PointCloud,
    percentile: float,
):
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    max_distance = float(np.percentile(distances, percentile))

    filtered_pcd = remover.filter_by_distance_from_center(
        pcd, max_distance=max_distance
    )

    return filtered_pcd, {
        "center": center.tolist(),
        "distance_percentile": float(percentile),
        "max_distance": max_distance,
    }


def run_single_variant(name: str, params: dict):
    print(f"\n{'=' * 60}")
    print(f"运行参数组: {name}")
    print(json.dumps(params, ensure_ascii=False, indent=2))
    print(f"{'=' * 60}")

    remover = SimpleBackgroundRemoval()
    remover.plane_threshold = params["plane_threshold"]

    pcd = remover.load_point_cloud(str(INPUT_PATH))
    remover.analyze_point_cloud(pcd)

    pcd_filtered = remover.remove_ground_and_ceiling(
        pcd,
        min_percentile=params["z_min_percentile"],
        max_percentile=params["z_max_percentile"],
    )

    pcd_no_planes, plane_stats = _remove_large_planes_with_stats(
        remover,
        pcd_filtered,
        threshold=params["plane_threshold"],
        min_points=params["plane_min_points"],
    )

    pcd_distance_filtered, distance_stats = _filter_by_distance_percentile(
        remover,
        pcd_no_planes,
        percentile=params["distance_percentile"],
    )

    pcd_final = remover.remove_outliers(pcd_distance_filtered)

    output_path = OUTPUT_DIR / f"left1_background_no_background_{name}.ply"
    o3d.io.write_point_cloud(str(output_path), pcd_final)

    final_count = len(pcd_final.points)
    original_count = len(pcd.points)
    retention_ratio = float(final_count / original_count * 100.0)

    print("\n=== 最终结果 ===")
    print(f"原始点数: {original_count}")
    print(f"最终点数: {final_count}")
    print(f"保留比例: {retention_ratio:.1f}%")
    print(f"结果已保存至: {output_path}")

    return {
        "name": name,
        "input_path": str(INPUT_PATH),
        "output_path": str(output_path),
        "params": params,
        "original_count": int(original_count),
        "after_z_filter_count": int(len(pcd_filtered.points)),
        "after_plane_filter_count": int(len(pcd_no_planes.points)),
        "after_distance_filter_count": int(len(pcd_distance_filtered.points)),
        "final_count": int(final_count),
        "retention_ratio_percent": retention_ratio,
        "plane_stats": plane_stats,
        "distance_stats": distance_stats,
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入点云不存在: {INPUT_PATH}")

    summary = {}
    for name, params in PARAMETER_SETS.items():
        summary[name] = run_single_variant(name, params)

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n对比摘要已保存至: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
