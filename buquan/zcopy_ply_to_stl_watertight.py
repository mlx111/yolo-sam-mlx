#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert z-copy point cloud to a watertight STL solid.

Pipeline:
1) Voxelize point cloud into occupancy grid.
2) Apply 3D morphology (closing/opening) and fill interior.
3) Extract mesh from solid voxels (marching cubes).
4) Smooth/clean/simplify and enforce watertight fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from trimesh.voxel import ops as voxel_ops


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _import_open3d():
    try:
        import open3d as o3d  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import open3d.") from exc
    return o3d


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert zcopy PLY to watertight STL using voxel solid reconstruction.")
    parser.add_argument(
        "--input-ply",
        default=os.path.join(SCRIPT_DIR, "outputs_inputs_to_zcopy_final", "pear_zcopy_filled_aligned.ply"),
        help="Input zcopy point cloud.",
    )
    parser.add_argument("--stl-out", default="", help="Output STL path. Default: <input>_watertight_voxel.stl")
    parser.add_argument("--report-out", default="", help="Output report path. Default: <input>_watertight_voxel_report.json")
    parser.add_argument("--mesh-ply-out", default="", help="Optional output mesh PLY path.")

    parser.add_argument("--voxel-mm", type=float, default=1.2, help="Voxel size for occupancy reconstruction.")
    parser.add_argument("--padding-vox", type=int, default=2, help="Padding voxels around occupancy grid.")
    parser.add_argument("--close-iters", type=int, default=2, help="Binary closing iterations.")
    parser.add_argument("--open-iters", type=int, default=0, help="Binary opening iterations.")

    parser.add_argument("--smooth-iters", type=int, default=8, help="Taubin smoothing iterations.")
    parser.add_argument("--smooth-lambda", type=float, default=0.5, help="Taubin smoothing lambda.")
    parser.add_argument("--target-faces", type=int, default=20000, help="Target faces after decimation; 0 disables.")
    parser.add_argument(
        "--max-split-faces",
        type=int,
        default=300000,
        help="Skip component split step when face count exceeds this threshold.",
    )
    parser.add_argument("--fallback", choices=["convex_hull", "none"], default="convex_hull")
    parser.add_argument("--sample-points", type=int, default=30000, help="Surface samples for fit metrics.")
    return parser.parse_args()


def load_points(ply_path: str) -> np.ndarray:
    o3d = _import_open3d()
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Input PLY not found: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError(f"Invalid or empty point cloud: {ply_path}")
    return pts


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    out.remove_unreferenced_vertices()
    out.update_faces(out.unique_faces())
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


def keep_largest_component(vol: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, num = ndimage.label(vol, structure=structure)
    if num <= 1:
        return vol
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    winner = int(np.argmax(counts))
    return labels == winner


def build_solid_voxels(
    points: np.ndarray,
    voxel_m: float,
    padding_vox: int,
    close_iters: int,
    open_iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pad = max(1, int(padding_vox))
    origin = points.min(axis=0) - voxel_m * float(pad)
    idx = np.floor((points - origin.reshape(1, 3)) / voxel_m).astype(np.int32)
    idx = np.maximum(idx, 0)
    max_idx = idx.max(axis=0)
    shape = tuple((max_idx + 1 + pad).tolist())

    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    structure = ndimage.generate_binary_structure(3, 1)
    solid = occ
    if close_iters > 0:
        solid = ndimage.binary_closing(solid, structure=structure, iterations=int(close_iters))
    if open_iters > 0:
        solid = ndimage.binary_opening(solid, structure=structure, iterations=int(open_iters))
    solid = ndimage.binary_fill_holes(solid)
    solid = keep_largest_component(solid)
    return solid, origin


def mesh_from_voxels(solid: np.ndarray, origin: np.ndarray, voxel_m: float) -> trimesh.Trimesh:
    if not np.any(solid):
        raise RuntimeError("Solid occupancy is empty after morphology.")

    try:
        # Use marching cubes to avoid generating an excessively dense multibox mesh.
        mesh = voxel_ops.matrix_to_marching_cubes(solid, pitch=float(voxel_m))
        mesh.apply_translation(origin.astype(np.float64))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to build mesh from occupied voxels.") from exc
    return clean_mesh(mesh)


def postprocess_mesh(
    mesh: trimesh.Trimesh,
    smooth_iters: int,
    smooth_lambda: float,
    target_faces: int,
    max_split_faces: int,
) -> trimesh.Trimesh:
    out = clean_mesh(mesh)

    if smooth_iters > 0:
        try:
            trimesh.smoothing.filter_taubin(
                out,
                lamb=float(smooth_lambda),
                nu=-0.53,
                iterations=int(smooth_iters),
            )
        except Exception:  # noqa: BLE001
            pass
        out = clean_mesh(out)

    # `split` can be very expensive on dense meshes; skip it above a safe threshold.
    if len(out.faces) <= int(max_split_faces):
        parts = out.split(only_watertight=False)
        if parts:
            out = max(parts, key=lambda m: len(m.faces))
            out = clean_mesh(out)

    if target_faces > 0 and len(out.faces) > int(target_faces):
        try:
            out = out.simplify_quadric_decimation(int(target_faces))
            out = clean_mesh(out)
        except Exception:  # noqa: BLE001
            pass

    return out


def enforce_watertight(mesh: trimesh.Trimesh, points: np.ndarray, fallback: str) -> Tuple[trimesh.Trimesh, bool]:
    out = clean_mesh(mesh)
    fallback_used = False

    if not out.is_watertight:
        try:
            trimesh.repair.fill_holes(out)
            out = clean_mesh(out)
        except Exception:  # noqa: BLE001
            pass

    if (not out.is_watertight or not out.is_volume) and fallback == "convex_hull":
        hull = trimesh.points.PointCloud(points).convex_hull
        out = clean_mesh(hull)
        fallback_used = True

    return out, fallback_used


def fit_metrics(mesh: trimesh.Trimesh, points: np.ndarray, sample_points: int) -> Dict[str, float]:
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return {"mean_m": float("inf"), "p90_m": float("inf")}
    if int(sample_points) <= 0:
        return {"mean_m": float("nan"), "p90_m": float("nan")}

    n = int(np.clip(sample_points, 1000, 100000))
    sampled, _ = trimesh.sample.sample_surface(mesh, n)
    tree_mesh = cKDTree(sampled)
    tree_pts = cKDTree(points)
    d_pm, _ = tree_mesh.query(points, k=1)
    d_mp, _ = tree_pts.query(sampled, k=1)
    both = np.concatenate([d_pm, d_mp], axis=0)
    return {
        "mean_m": float(np.mean(both)),
        "p90_m": float(np.percentile(both, 90)),
    }


def default_paths(input_ply: str, stl_out: str, report_out: str) -> Tuple[str, str]:
    base, _ = os.path.splitext(os.path.abspath(input_ply))
    if not stl_out:
        stl_out = base + "_watertight_voxel.stl"
    if not report_out:
        report_out = base + "_watertight_voxel_report.json"
    return os.path.abspath(stl_out), os.path.abspath(report_out)


def finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(float(x)) else None


def main() -> int:
    args = parse_args()
    input_ply = os.path.abspath(args.input_ply)
    stl_out, report_out = default_paths(input_ply, args.stl_out, args.report_out)
    mesh_ply_out = os.path.abspath(args.mesh_ply_out) if args.mesh_ply_out else ""

    voxel_m = max(1e-5, float(args.voxel_mm) / 1000.0)
    points = load_points(input_ply)
    input_bbox = points.max(axis=0) - points.min(axis=0)

    solid, origin = build_solid_voxels(
        points=points,
        voxel_m=voxel_m,
        padding_vox=max(1, int(args.padding_vox)),
        close_iters=max(0, int(args.close_iters)),
        open_iters=max(0, int(args.open_iters)),
    )
    mesh = mesh_from_voxels(solid, origin, voxel_m)
    mesh = postprocess_mesh(
        mesh=mesh,
        smooth_iters=max(0, int(args.smooth_iters)),
        smooth_lambda=float(args.smooth_lambda),
        target_faces=max(0, int(args.target_faces)),
        max_split_faces=max(0, int(args.max_split_faces)),
    )
    mesh, fallback_used = enforce_watertight(mesh, points, str(args.fallback))

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise RuntimeError("Empty mesh generated.")

    fit = fit_metrics(mesh, points, int(args.sample_points))
    mesh_bbox = mesh.extents
    bbox_rel = np.abs(mesh_bbox - input_bbox) / np.maximum(input_bbox, 1e-9)

    ensure_parent(stl_out)
    mesh.export(stl_out)
    if mesh_ply_out:
        ensure_parent(mesh_ply_out)
        mesh.export(mesh_ply_out)

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "input_ply": input_ply,
        "output_stl": stl_out,
        "output_mesh_ply": mesh_ply_out,
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "volume_m3": finite_or_none(float(mesh.volume) if mesh.is_volume else float("nan")),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "bbox_input_m": input_bbox.tolist(),
        "bbox_mesh_m": mesh_bbox.tolist(),
        "bbox_rel_error": bbox_rel.tolist(),
        "fit_mean_mm": finite_or_none(fit["mean_m"] * 1000.0),
        "fit_p90_mm": finite_or_none(fit["p90_m"] * 1000.0),
        "fallback_used": bool(fallback_used),
        "config": {
            "voxel_mm": float(args.voxel_mm),
            "padding_vox": int(args.padding_vox),
            "close_iters": int(args.close_iters),
            "open_iters": int(args.open_iters),
            "smooth_iters": int(args.smooth_iters),
            "smooth_lambda": float(args.smooth_lambda),
            "target_faces": int(args.target_faces),
            "max_split_faces": int(args.max_split_faces),
            "fallback": str(args.fallback),
            "sample_points": int(args.sample_points),
        },
    }

    ensure_parent(report_out)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] stl: {stl_out}")
    print(f"[INFO] report: {report_out}")
    print(
        f"[INFO] watertight={report['watertight']} volume={report['is_volume']} "
        f"faces={report['mesh_faces']} fit_mean_mm={report['fit_mean_mm']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
