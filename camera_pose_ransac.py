from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


@dataclass
class PlaneCandidate:
    index: int
    inlier_count: int
    inlier_ratio: float
    plane_model: list[float]
    center: list[float]
    normal: list[float]
    major_axis: list[float]
    minor_axis: list[float]
    extents: list[float]
    score: float


@dataclass
class RawCameraPose:
    point_cloud_path: str
    selected_plane_index: int
    plane_candidates: list[dict[str, Any]]
    rotation_matrix: list[list[float]]
    translation_reference_in_camera: list[float]
    camera_origin_in_reference: list[float]
    euler_xyz_deg: list[float]


def _load_point_cloud(path: str | Path, voxel_size: float = 0.0) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


def _segment_plane_candidates(
    points: np.ndarray,
    max_planes: int = 5,
    distance_threshold: float = 0.01,
    min_ratio: float = 0.02,
    num_iterations: int = 1000,
) -> list[PlaneCandidate]:
    remaining = o3d.geometry.PointCloud()
    remaining.points = o3d.utility.Vector3dVector(points)
    total_points = len(points)
    candidates: list[PlaneCandidate] = []

    for plane_index in range(max_planes):
        if len(remaining.points) < 100:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations,
        )
        if not inliers:
            break

        inlier_ratio = len(inliers) / total_points
        if inlier_ratio < min_ratio:
            break

        inlier_pcd = remaining.select_by_index(inliers)
        inlier_points = np.asarray(inlier_pcd.points, dtype=float)
        center = inlier_points.mean(axis=0)

        normal = np.asarray(plane_model[:3], dtype=float)
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        # Flip the normal so it points back to the camera origin.
        if np.dot(normal, center) > 0:
            normal = -normal

        projected = inlier_points - np.outer(inlier_points @ normal, normal)
        projected -= projected.mean(axis=0, keepdims=True)
        cov = np.cov(projected.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        major = eigvecs[:, 0]
        major = major - normal * float(np.dot(major, normal))
        if np.linalg.norm(major) < 1e-9:
            major = np.array([1.0, 0.0, 0.0], dtype=float)
        major = major / np.linalg.norm(major)

        minor = np.cross(normal, major)
        minor = minor / max(np.linalg.norm(minor), 1e-12)
        major = np.cross(minor, normal)
        major = major / max(np.linalg.norm(major), 1e-12)

        coords_major = projected @ major
        coords_minor = projected @ minor
        extents = [
            float(coords_major.max() - coords_major.min()),
            float(coords_minor.max() - coords_minor.min()),
        ]
        score = float(len(inliers) * max(extents[0], 1e-6) * max(extents[1], 1e-6))

        candidates.append(
            PlaneCandidate(
                index=plane_index,
                inlier_count=len(inliers),
                inlier_ratio=float(inlier_ratio),
                plane_model=[float(x) for x in plane_model],
                center=[float(x) for x in center],
                normal=[float(x) for x in normal],
                major_axis=[float(x) for x in major],
                minor_axis=[float(x) for x in minor],
                extents=extents,
                score=score,
            )
        )

        remaining = remaining.select_by_index(inliers, invert=True)

    return candidates


def _reference_frame_from_plane(candidate: PlaneCandidate) -> tuple[np.ndarray, np.ndarray]:
    x_axis_cam = np.asarray(candidate.major_axis, dtype=float)
    z_axis_cam = np.asarray(candidate.normal, dtype=float)
    y_axis_cam = np.cross(z_axis_cam, x_axis_cam)
    y_axis_cam = y_axis_cam / max(np.linalg.norm(y_axis_cam), 1e-12)
    x_axis_cam = np.cross(y_axis_cam, z_axis_cam)
    x_axis_cam = x_axis_cam / max(np.linalg.norm(x_axis_cam), 1e-12)

    # Columns are reference-frame axes expressed in camera coordinates.
    reference_axes_in_camera = np.column_stack([x_axis_cam, y_axis_cam, z_axis_cam])
    rotation_camera_to_reference = reference_axes_in_camera.T
    translation_reference_in_camera = np.asarray(candidate.center, dtype=float)
    return rotation_camera_to_reference, translation_reference_in_camera


def estimate_raw_camera_pose(
    point_cloud_path: str | Path,
    *,
    voxel_size: float = 0.01,
    max_planes: int = 5,
    distance_threshold: float = 0.01,
    min_ratio: float = 0.02,
) -> RawCameraPose:
    points = _load_point_cloud(point_cloud_path, voxel_size=voxel_size)
    candidates = _segment_plane_candidates(
        points,
        max_planes=max_planes,
        distance_threshold=distance_threshold,
        min_ratio=min_ratio,
    )
    if not candidates:
        raise ValueError(f"No valid plane candidate found in {point_cloud_path}")

    selected = max(candidates, key=lambda item: item.score)
    rotation_camera_to_reference, translation_reference_in_camera = _reference_frame_from_plane(selected)
    camera_origin_in_reference = -rotation_camera_to_reference @ translation_reference_in_camera
    euler_xyz_deg = Rotation.from_matrix(rotation_camera_to_reference).as_euler("xyz", degrees=True)

    return RawCameraPose(
        point_cloud_path=str(Path(point_cloud_path).resolve()),
        selected_plane_index=selected.index,
        plane_candidates=[asdict(item) for item in candidates],
        rotation_matrix=rotation_camera_to_reference.tolist(),
        translation_reference_in_camera=translation_reference_in_camera.tolist(),
        camera_origin_in_reference=camera_origin_in_reference.tolist(),
        euler_xyz_deg=[float(x) for x in euler_xyz_deg],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a raw camera pose from a background point cloud.")
    parser.add_argument("--point-cloud", required=True, help="Path to the global/background point cloud.")
    parser.add_argument("--json-out", help="Optional output JSON path.")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--max-planes", type=int, default=5)
    parser.add_argument("--distance-threshold", type=float, default=0.01)
    parser.add_argument("--min-ratio", type=float, default=0.02)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    pose = estimate_raw_camera_pose(
        args.point_cloud,
        voxel_size=args.voxel_size,
        max_planes=args.max_planes,
        distance_threshold=args.distance_threshold,
        min_ratio=args.min_ratio,
    )
    payload = json.dumps(asdict(pose), ensure_ascii=False, indent=2)
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
