from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class CompletionConfig:
    voxel_size_m: float = 0.0015
    stat_k: int = 20
    stat_std_ratio: float = 2.0
    radius_m: float = 0.006
    radius_min_neighbors: int = 6
    cluster_radius_m: float = 0.005
    tail_bin_m: float = 0.0012
    tail_min_support: int = 8
    tail_max_remove_ratio: float = 0.03
    center_mode: str = "mean"
    align_axis: str = "minor"
    copy_source: str = "cleaned_raw"
    copy_times: int = 10
    copy_step_m: float = 0.0025
    copy_direction: str = "both"
    copy_voxel_m: float = 0.0008


def voxel_downsample_np(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    if voxel <= 0.0 or len(points) == 0:
        return points, colors

    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    out_points = points[idx]
    out_colors = colors[idx] if colors is not None else None
    return out_points, out_colors


def statistical_outlier_filter(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    if len(points) < max(50, int(k) + 1):
        return points

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(int(k) + 1, len(points)))
    mean_k = dists[:, 1:].mean(axis=1)
    mu = float(mean_k.mean())
    sigma = float(mean_k.std() + 1e-9)
    keep = mean_k <= (mu + float(std_ratio) * sigma)
    filtered = points[keep]
    if len(filtered) < max(100, int(0.35 * len(points))):
        return points
    return filtered


def radius_outlier_filter(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    if len(points) < 100 or radius <= 0:
        return points

    tree = cKDTree(points)
    counts = np.array([len(v) - 1 for v in tree.query_ball_point(points, r=float(radius))], dtype=np.int32)
    keep = counts >= int(min_neighbors)
    filtered = points[keep]
    if len(filtered) < max(100, int(0.35 * len(points))):
        return points
    return filtered


def largest_connected_component(
    points: np.ndarray,
    radius: float,
    min_keep_ratio: float = 0.25,
) -> tuple[np.ndarray, Dict[str, object]]:
    if len(points) < 50 or radius <= 0:
        return points, {"applied": False, "reason": "too_few_points_or_nonpositive_radius"}

    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=float(radius))
    visited = np.zeros(len(points), dtype=bool)
    best: list[int] = []

    for i in range(len(points)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp

    if not best:
        return points, {"applied": False, "reason": "empty_components"}

    min_keep = max(100, int(float(min_keep_ratio) * len(points)))
    if len(best) < min_keep:
        return points, {
            "applied": False,
            "reason": "largest_component_too_small",
            "largest_component_size": int(len(best)),
        }

    out = points[np.asarray(best, dtype=np.int64)]
    return out, {
        "applied": True,
        "largest_component_size": int(len(best)),
        "input_points": int(len(points)),
    }


def trim_sparse_tail_along_major_axis(
    points: np.ndarray,
    bin_m: float,
    min_support: int,
    max_remove_ratio: float,
) -> tuple[np.ndarray, Dict[str, object]]:
    if len(points) < 200 or bin_m <= 0:
        return points, {"applied": False, "reason": "too_few_points_or_nonpositive_bin"}

    center = points.mean(axis=0)
    centered = points - center.reshape(1, 3)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ axis
    p_min, p_max = float(np.min(proj)), float(np.max(proj))
    if (p_max - p_min) < (3.0 * float(bin_m)):
        return points, {"applied": False, "reason": "projection_range_too_small"}

    bins = np.floor((proj - p_min) / float(bin_m)).astype(np.int64)
    n_bins = int(np.max(bins)) + 1
    counts = np.bincount(bins, minlength=n_bins)

    max_remove = max(1, int(float(max_remove_ratio) * len(points)))
    total_removed = 0
    left, right = 0, n_bins - 1

    while left < right and counts[left] < int(min_support) and (total_removed + int(counts[left])) <= max_remove:
        total_removed += int(counts[left])
        left += 1
    while left < right and counts[right] < int(min_support) and (total_removed + int(counts[right])) <= max_remove:
        total_removed += int(counts[right])
        right -= 1

    if left == 0 and right == (n_bins - 1):
        return points, {"applied": False, "reason": "no_sparse_tail_detected"}

    keep = (bins >= left) & (bins <= right)
    out = points[keep]
    if len(out) < max(150, int(0.5 * len(points))):
        return points, {"applied": False, "reason": "trim_too_aggressive", "candidate_points": int(len(out))}

    keep_min = p_min + float(left) * float(bin_m)
    keep_max = p_min + float(right + 1) * float(bin_m)
    return out, {
        "applied": True,
        "input_points": int(len(points)),
        "output_points": int(len(out)),
        "removed_points": int(len(points) - len(out)),
        "removed_ratio": float((len(points) - len(out)) / max(1, len(points))),
        "left_trim_mm": float(max(0.0, keep_min - p_min) * 1000.0),
        "right_trim_mm": float(max(0.0, p_max - keep_max) * 1000.0),
        "tail_bin_mm": float(bin_m * 1000.0),
        "tail_min_support": int(min_support),
    }


def center_points(points: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "bbox":
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    else:
        center = points.mean(axis=0)
    return points - center, center


def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    a = vec_from / (np.linalg.norm(vec_from) + 1e-12)
    b = vec_to / (np.linalg.norm(vec_to) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v /= np.linalg.norm(v) + 1e-12
        vx = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * vx @ vx

    s = np.linalg.norm(v)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s + 1e-12))


def align_points_to_z(points_centered: np.ndarray, align_axis: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis_map = {"major": 0, "middle": 1, "minor": 2}
    idx = axis_map[align_axis]
    axis = eigvecs[:, idx]
    if float(np.dot(axis, np.array([0.0, 0.0, 1.0]))) < 0:
        axis = -axis
    rot = rotation_matrix_from_vectors(axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    aligned = (rot @ points_centered.T).T
    return aligned, rot, eigvals.tolist()


def build_copy_shifts(times: int, step_m: float, direction: str) -> list[float]:
    n = max(0, int(times))
    step = float(step_m)
    if direction == "both":
        vals = [0.0]
        for i in range(1, n + 1):
            vals.append(float(i) * step)
            vals.append(-float(i) * step)
        return vals
    if direction == "pos":
        return [0.0] + [float(i) * step for i in range(1, n + 1)]
    return [0.0] + [-float(i) * step for i in range(1, n + 1)]


def copy_fill_along_z(points_aligned: np.ndarray, times: int, step_m: float, direction: str) -> tuple[np.ndarray, list[float]]:
    shifts = build_copy_shifts(times, step_m, direction)
    layers = []
    for s in shifts:
        p = points_aligned.copy()
        p[:, 2] += s
        layers.append(p)
    return np.vstack(layers), shifts


def point_cloud_bounds(points: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty Nx3 array")
    return {
        "min": np.min(points, axis=0),
        "max": np.max(points, axis=0),
        "extent": np.max(points, axis=0) - np.min(points, axis=0),
    }


def make_point_cloud(points: np.ndarray, colors: np.ndarray | None = None):
    import open3d as o3d  # local import to keep this module lightweight for non-visual use

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    if colors is None:
        colors = np.zeros((len(points), 3), dtype=np.float64)
    cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))
    return cloud


@dataclass
class CompletionResult:
    completed_points: np.ndarray
    completed_colors: np.ndarray | None
    center: np.ndarray
    rotation_to_z: np.ndarray
    report: Dict[str, object]


def complete_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    cfg: CompletionConfig | None = None,
) -> CompletionResult:
    cfg = cfg or CompletionConfig()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")

    raw_points = points

    raw_base = statistical_outlier_filter(raw_points, cfg.stat_k, cfg.stat_std_ratio)
    raw_base = radius_outlier_filter(raw_base, cfg.radius_m, cfg.radius_min_neighbors)
    clustered_raw, cluster_stats = largest_connected_component(raw_base, cfg.cluster_radius_m)
    cleaned_raw, tail_stats = trim_sparse_tail_along_major_axis(
        clustered_raw,
        bin_m=cfg.tail_bin_m,
        min_support=cfg.tail_min_support,
        max_remove_ratio=cfg.tail_max_remove_ratio,
    )
    if len(cleaned_raw) == 0:
        cleaned_raw = clustered_raw if len(clustered_raw) > 0 else raw_points

    processed = voxel_downsample_np(cleaned_raw, None, cfg.voxel_size_m)[0]
    processed = statistical_outlier_filter(processed, cfg.stat_k, cfg.stat_std_ratio)
    processed = radius_outlier_filter(processed, cfg.radius_m, cfg.radius_min_neighbors)
    if len(processed) == 0:
        processed = cleaned_raw

    processed_centered, center = center_points(processed, cfg.center_mode)
    processed_aligned, rot, eigvals = align_points_to_z(processed_centered, cfg.align_axis)

    raw_centered = raw_points - center.reshape(1, 3)
    raw_aligned = (rot @ raw_centered.T).T
    cleaned_raw_centered = cleaned_raw - center.reshape(1, 3)
    cleaned_raw_aligned = (rot @ cleaned_raw_centered.T).T
    if cfg.copy_source == "raw":
        copy_base = raw_aligned
    elif cfg.copy_source == "cleaned_raw":
        copy_base = cleaned_raw_aligned
    else:
        copy_base = processed_aligned

    copied_aligned, copy_shifts = copy_fill_along_z(
        points_aligned=copy_base,
        times=cfg.copy_times,
        step_m=cfg.copy_step_m,
        direction=cfg.copy_direction,
    )
    filled_points_before_voxel = int(len(copied_aligned))
    copied_aligned, _ = voxel_downsample_np(copied_aligned, None, cfg.copy_voxel_m)
    completed_points = (rot.T @ copied_aligned.T).T + center.reshape(1, 3)

    report: Dict[str, object] = {
        "counts": {
            "raw_points": int(len(raw_points)),
            "raw_after_stat_radius": int(len(raw_base)),
            "raw_largest_cluster_points": int(len(clustered_raw)),
            "raw_tail_trimmed_points": int(len(cleaned_raw)),
            "preprocessed_points": int(len(processed)),
            "aligned_raw_points": int(len(raw_aligned)),
            "aligned_cleaned_raw_points": int(len(cleaned_raw_aligned)),
            "aligned_processed_points": int(len(processed_aligned)),
            "copy_base_points": int(len(copy_base)),
            "filled_points_before_voxel": filled_points_before_voxel,
            "filled_points": int(len(copied_aligned)),
            "final_completed_points": int(len(completed_points)),
        },
        "geometry": {
            "center_translation_xyz_m": center.tolist(),
            "rotation_to_z_3x3": rot.tolist(),
            "pca_eigenvalues": eigvals,
            "copy_shifts_z_m": [float(v) for v in copy_shifts],
            "voxel_size_m": float(cfg.voxel_size_m),
        },
        "outlier_filter": {
            "cluster_filter": cluster_stats,
            "tail_trim": tail_stats,
        },
    }
    return CompletionResult(
        completed_points=completed_points.astype(np.float32),
        completed_colors=None,
        center=center.astype(np.float32),
        rotation_to_z=rot.astype(np.float32),
        report=report,
    )
