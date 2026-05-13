#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline:
1) Build object point cloud from selected camera depth + object mask.
2) Apply stronger outlier removal on raw cloud.
3) Denoise/center and rotate selected PCA axis to +Z.
4) Strictly copy points along Z (no interpolation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _import_open3d():
    try:
        import open3d as o3d  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import open3d.") from exc
    return o3d


@dataclass
class Config:
    object_name: str
    input_root: str
    calib_pdf: str
    extrinsics_json: str
    camera: str = "left"
    left_depth_rel: str = "dleft001.png"
    left_mask_tpl: str = "left_mask_{obj}.png"
    right_depth_rel: str = "dright001.png"
    right_mask_tpl: str = "right_mask_{obj}.png"
    depth_scale: float = 1000.0
    depth_trunc_m: float = 6.0
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
    copy_source: str = "cleaned_raw"  # raw | cleaned_raw | processed
    copy_times: int = 12
    copy_step_m: float = 0.0005
    copy_direction: str = "neg"  # neg | pos | both
    copy_voxel_m: float = 0.0008
    output_dir: str = os.path.join(SCRIPT_DIR, "outputs_inputs_to_zcopy")
    save_report: bool = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_depth(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    if img.dtype != np.uint16:
        raise ValueError(f"Depth image must be uint16, got {img.dtype}: {path}")
    return img


def read_mask(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask image: {path}")
    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def save_points(path: str, points: np.ndarray) -> None:
    o3d = _import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def voxel_downsample_np(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def statistical_outlier_filter(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    if len(points) < max(50, k + 1):
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(k + 1, len(points)))
    mean_k = dists[:, 1:].mean(axis=1)
    mu, sigma = float(mean_k.mean()), float(mean_k.std() + 1e-9)
    keep = mean_k <= (mu + std_ratio * sigma)
    filtered = points[keep]
    return filtered if len(filtered) > max(100, int(0.35 * len(points))) else points


def radius_outlier_filter(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    if len(points) < 100:
        return points
    tree = cKDTree(points)
    counts = np.array([len(v) - 1 for v in tree.query_ball_point(points, r=radius)], dtype=np.int32)
    keep = counts >= min_neighbors
    filtered = points[keep]
    return filtered if len(filtered) > max(100, int(0.35 * len(points))) else points


def largest_connected_component(
    points: np.ndarray,
    radius: float,
    min_keep_ratio: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(points) < 50 or radius <= 0:
        return points, {"applied": False, "reason": "too_few_points_or_nonpositive_radius"}

    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=radius)
    visited = np.zeros(len(points), dtype=bool)
    best: List[int] = []

    for i in range(len(points)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: List[int] = []
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

    min_keep = max(100, int(min_keep_ratio * len(points)))
    if len(best) < min_keep:
        return points, {"applied": False, "reason": "largest_component_too_small", "largest_component_size": int(len(best))}

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
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(points) < 200 or bin_m <= 0:
        return points, {"applied": False, "reason": "too_few_points_or_nonpositive_bin"}

    center = points.mean(axis=0)
    centered = points - center.reshape(1, 3)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ axis
    p_min, p_max = float(np.min(proj)), float(np.max(proj))
    if (p_max - p_min) < (3.0 * bin_m):
        return points, {"applied": False, "reason": "projection_range_too_small"}

    bins = np.floor((proj - p_min) / bin_m).astype(np.int64)
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

    keep_min = p_min + float(left) * bin_m
    keep_max = p_min + float(right + 1) * bin_m
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


def center_points(points: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
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


def align_points_to_z(points_centered: np.ndarray, align_axis: str) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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


def _extract_text_from_pdf(pdf_path: str) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_first_4_intrinsics(block: str) -> Tuple[float, float, float, float]:
    m_fx = re.search(r"焦距:\s*fx\s*=\s*([-\d.]+)\s*,\s*fy\s*=\s*([-\d.]+)", block)
    m_cx = re.search(r"主点:\s*cx\s*=\s*([-\d.]+)\s*,\s*cy\s*=\s*([-\d.]+)", block)
    if not m_fx or not m_cx:
        raise ValueError("Failed to parse fx/fy/cx/cy from calibration PDF.")
    fx, fy = float(m_fx.group(1)), float(m_fx.group(2))
    cx, cy = float(m_cx.group(1)), float(m_cx.group(2))
    return fx, fy, cx, cy


def _floats_from_text(text: str) -> List[float]:
    normalized = re.sub(r"([+-])\s+(\d)", r"\1\2", text)
    return [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)]


def load_calibration_from_pdf(pdf_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Calibration PDF not found: {pdf_path}")
    text = _extract_text_from_pdf(pdf_path)
    left_block_m = re.search(r"左内参(.*?)右内参", text, flags=re.S)
    right_block_m = re.search(r"右内参(.*?)外参", text, flags=re.S)
    if not left_block_m or not right_block_m:
        raise ValueError("Failed to locate left/right intrinsic sections in calibration PDF.")

    left_fx, left_fy, left_cx, left_cy = _parse_first_4_intrinsics(left_block_m.group(1))
    right_fx, right_fy, right_cx, right_cy = _parse_first_4_intrinsics(right_block_m.group(1))
    k_left = np.array([[left_fx, 0.0, left_cx], [0.0, left_fy, left_cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    k_right = np.array([[right_fx, 0.0, right_cx], [0.0, right_fy, right_cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    r_block_m = re.search(r"旋转矩阵R】(.*?)【平移向量T", text, flags=re.S)
    t_block_m = re.search(r"平移向量T（单位：mm）】(.*)", text, flags=re.S)
    if not r_block_m or not t_block_m:
        raise ValueError("Failed to parse extrinsic R/T blocks in calibration PDF.")
    r_vals = _floats_from_text(r_block_m.group(1))
    t_vals = _floats_from_text(t_block_m.group(1))
    if len(r_vals) < 9 or len(t_vals) < 3:
        raise ValueError("Parsed calibration values are incomplete.")
    r_pdf = np.array(r_vals[:9], dtype=np.float64).reshape(3, 3)
    t_pdf_m = np.array(t_vals[:3], dtype=np.float64) / 1000.0

    return {
        "k_left": k_left,
        "k_right": k_right,
        "r_pdf": r_pdf,
        "t_pdf_m": t_pdf_m,
    }


def load_refined_extrinsics(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    t = np.array(data.get("transform_right_to_left"), dtype=np.float64)
    return t if t.shape == (4, 4) else None


def build_points_from_left_depth_and_mask(
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    k_left: np.ndarray,
    depth_scale: float,
    depth_trunc_m: float,
) -> np.ndarray:
    if depth_u16.shape[:2] != mask01.shape[:2]:
        raise ValueError("Depth and mask size mismatch.")
    z_m = depth_u16.astype(np.float64) / float(depth_scale)
    valid = (mask01 > 0) & (depth_u16 > 0) & np.isfinite(z_m) & (z_m > 0.0) & (z_m <= float(depth_trunc_m))
    if int(np.sum(valid)) == 0:
        raise ValueError("No valid object pixels after applying mask/depth filters.")

    v, u = np.where(valid)
    z = z_m[v, u]
    fx, fy = float(k_left[0, 0]), float(k_left[1, 1])
    cx, cy = float(k_left[0, 2]), float(k_left[1, 2])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float64)


def build_copy_shifts(times: int, step_m: float, direction: str) -> List[float]:
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


def copy_fill_along_z(points_aligned: np.ndarray, times: int, step_m: float, direction: str) -> Tuple[np.ndarray, List[float]]:
    shifts = build_copy_shifts(times, step_m, direction)
    layers = []
    for s in shifts:
        p = points_aligned.copy()
        p[:, 2] += s
        layers.append(p)
    return np.vstack(layers), shifts


def run(cfg: Config) -> Dict[str, object]:
    ensure_dir(cfg.output_dir)
    calib = load_calibration_from_pdf(cfg.calib_pdf)
    t_refined = load_refined_extrinsics(cfg.extrinsics_json)

    if cfg.camera == "right":
        depth_rel = cfg.right_depth_rel
        mask_tpl = cfg.right_mask_tpl
        selected_k = calib["k_right"]
        selected_k_name = "k_right"
    else:
        depth_rel = cfg.left_depth_rel
        mask_tpl = cfg.left_mask_tpl
        selected_k = calib["k_left"]
        selected_k_name = "k_left"

    depth_path = os.path.join(cfg.input_root, depth_rel)
    mask_path = os.path.join(cfg.input_root, mask_tpl.format(obj=cfg.object_name))
    depth_u16 = read_depth(depth_path)
    mask01 = read_mask(mask_path, depth_u16.shape[:2])
    raw_points = build_points_from_left_depth_and_mask(
        depth_u16=depth_u16,
        mask01=mask01,
        k_left=selected_k,
        depth_scale=cfg.depth_scale,
        depth_trunc_m=cfg.depth_trunc_m,
    )

    # Strong outlier removal immediately after image->point cloud conversion.
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

    processed = voxel_downsample_np(cleaned_raw, cfg.voxel_size_m)
    processed = statistical_outlier_filter(processed, cfg.stat_k, cfg.stat_std_ratio)
    processed = radius_outlier_filter(processed, cfg.radius_m, cfg.radius_min_neighbors)
    if len(processed) == 0:
        processed = cleaned_raw
    processed_centered, center = center_points(processed, cfg.center_mode)
    processed_aligned, rot, eigvals = align_points_to_z(processed_centered, cfg.align_axis)

    # Apply the same centering+rotation to selectable base clouds.
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
    copied_aligned = voxel_downsample_np(copied_aligned, cfg.copy_voxel_m)

    out_filled = os.path.join(cfg.output_dir, f"{cfg.object_name}_zcopy_filled_aligned.ply")
    out_report = os.path.join(cfg.output_dir, f"{cfg.object_name}_pipeline_report.json")
    save_points(out_filled, copied_aligned)

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "inputs": {
            "camera_selected": cfg.camera,
            "selected_depth": depth_path,
            "selected_mask": mask_path,
            "left_depth": os.path.join(cfg.input_root, cfg.left_depth_rel),
            "left_mask": os.path.join(cfg.input_root, cfg.left_mask_tpl.format(obj=cfg.object_name)),
            "right_depth": os.path.join(cfg.input_root, cfg.right_depth_rel),
            "right_mask": os.path.join(cfg.input_root, cfg.right_mask_tpl.format(obj=cfg.object_name)),
            "calib_pdf": cfg.calib_pdf,
            "refined_extrinsics_json": cfg.extrinsics_json,
        },
        "calibration": {
            "k_left": calib["k_left"].tolist(),
            "k_right": calib["k_right"].tolist(),
            "selected_intrinsics_name": selected_k_name,
            "selected_intrinsics": selected_k.tolist(),
            "r_pdf": calib["r_pdf"].tolist(),
            "t_pdf_m": calib["t_pdf_m"].tolist(),
            "refined_transform_right_to_left_4x4": t_refined.tolist() if t_refined is not None else [],
        },
        "counts": {
            "mask_pixels": int(np.sum(mask01 > 0)),
            "raw_points": int(len(raw_points)),
            "raw_after_stat_radius": int(len(raw_base)),
            "raw_largest_cluster_points": int(len(clustered_raw)),
            "raw_tail_trimmed_points": int(len(cleaned_raw)),
            "preprocessed_points": int(len(processed)),
            "aligned_raw_points": int(len(raw_aligned)),
            "aligned_cleaned_raw_points": int(len(cleaned_raw_aligned)),
            "aligned_processed_points": int(len(processed_aligned)),
            "copy_base_points": int(len(copy_base)),
            "filled_points": int(len(copied_aligned)),
        },
        "outlier_filter": {
            "cluster_filter": cluster_stats,
            "tail_trim": tail_stats,
        },
        "geometry": {
            "center_translation_xyz_m": center.tolist(),
            "rotation_to_z_3x3": rot.tolist(),
            "pca_eigenvalues": eigvals,
            "copy_shifts_z_m": [float(v) for v in copy_shifts],
        },
        "outputs": {
            "zcopy_filled_aligned_ply": out_filled,
        },
    }
    if cfg.save_report:
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = out_report
    else:
        report["report_path"] = ""
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Inputs image -> preprocess -> align -> strict Z-copy fill.")
    parser.add_argument("--object", default="apple", help="Object name used in mask template.")
    parser.add_argument("--input-root", default=os.path.join(SCRIPT_DIR, "inputs"))
    parser.add_argument(
        "--calib-pdf",
        default=os.path.join(SCRIPT_DIR, "inputs", "双ob(1)", "双ob", "标定.pdf"),
        help="Calibration PDF path (required source for camera intrinsics).",
    )
    parser.add_argument(
        "--extrinsics-json",
        default=os.path.join(SCRIPT_DIR, "calib", "refined_stereo_extrinsics.json"),
        help="Optional refined right->left extrinsics json (for report only).",
    )
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--left-depth-rel", default="dleft001.png")
    parser.add_argument("--left-mask-tpl", default="left_mask_{obj}.png")
    parser.add_argument("--right-depth-rel", default="dright001.png")
    parser.add_argument("--right-mask-tpl", default="right_mask_{obj}.png")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc-m", type=float, default=6.0)

    parser.add_argument("--voxel-mm", type=float, default=1.5)
    parser.add_argument("--stat-k", type=int, default=20)
    parser.add_argument("--stat-std", type=float, default=2.0)
    parser.add_argument("--radius-mm", type=float, default=6.0)
    parser.add_argument("--radius-min-neighbors", type=int, default=6)
    parser.add_argument("--cluster-radius-mm", type=float, default=5.0, help="Radius for largest-component extraction.")
    parser.add_argument("--tail-bin-mm", type=float, default=1.2, help="Bin size for sparse-tail trimming on major axis.")
    parser.add_argument("--tail-min-support", type=int, default=8, help="Min points per tail bin to keep.")
    parser.add_argument(
        "--tail-max-remove-ratio",
        type=float,
        default=0.03,
        help="Upper bound of removed points during sparse-tail trimming.",
    )
    parser.add_argument("--center-mode", choices=["mean", "bbox"], default="mean")
    parser.add_argument("--align-axis", choices=["major", "middle", "minor"], default="minor")
    parser.add_argument("--copy-source", choices=["raw", "cleaned_raw", "processed"], default="cleaned_raw")

    parser.add_argument("--copy-times", type=int, default=12, help="Copy layers count on selected direction(s).")
    parser.add_argument("--copy-step-mm", type=float, default=0.5)
    parser.add_argument("--copy-direction", choices=["neg", "pos", "both"], default="neg")
    parser.add_argument("--copy-voxel-mm", type=float, default=0.8, help="Set 0 to disable dedup.")

    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "outputs_inputs_to_zcopy"))
    parser.add_argument("--no-report", action="store_true", help="Disable JSON report output.")
    args = parser.parse_args()

    return Config(
        object_name=str(args.object),
        input_root=os.path.abspath(args.input_root),
        calib_pdf=os.path.abspath(args.calib_pdf),
        extrinsics_json=os.path.abspath(args.extrinsics_json),
        camera=str(args.camera),
        left_depth_rel=str(args.left_depth_rel),
        left_mask_tpl=str(args.left_mask_tpl),
        right_depth_rel=str(args.right_depth_rel),
        right_mask_tpl=str(args.right_mask_tpl),
        depth_scale=float(args.depth_scale),
        depth_trunc_m=float(args.depth_trunc_m),
        voxel_size_m=max(0.0, float(args.voxel_mm) / 1000.0),
        stat_k=max(1, int(args.stat_k)),
        stat_std_ratio=float(args.stat_std),
        radius_m=max(0.0, float(args.radius_mm) / 1000.0),
        radius_min_neighbors=max(1, int(args.radius_min_neighbors)),
        cluster_radius_m=max(0.0, float(args.cluster_radius_mm) / 1000.0),
        tail_bin_m=max(1e-6, float(args.tail_bin_mm) / 1000.0),
        tail_min_support=max(1, int(args.tail_min_support)),
        tail_max_remove_ratio=float(np.clip(args.tail_max_remove_ratio, 0.0, 0.2)),
        center_mode=str(args.center_mode),
        align_axis=str(args.align_axis),
        copy_source=str(args.copy_source),
        copy_times=max(0, int(args.copy_times)),
        copy_step_m=max(0.0, float(args.copy_step_mm) / 1000.0),
        copy_direction=str(args.copy_direction),
        copy_voxel_m=max(0.0, float(args.copy_voxel_mm) / 1000.0),
        output_dir=os.path.abspath(args.output_dir),
        save_report=not bool(args.no_report),
    )


def main() -> int:
    cfg = parse_args()
    report = run(cfg)
    print(f"[INFO] counts: {report['counts']}")
    print(f"[INFO] outputs: {report['outputs']}")
    print(f"[INFO] report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
