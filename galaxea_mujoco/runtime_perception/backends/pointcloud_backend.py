from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np


DEFAULT_CAMERA_INTRINSICS = {"fx": 385.78, "fy": 385.22, "cx": 327.58, "cy": 238.69}

DEFAULT_CAMERA_EXTRINSICS = {
    "camera_to_tcp_pose": [0, 0, 0, -90 / 180 * pi, 0, -90 / 180 * pi],
    "tcp_pose": [150, -50, 570, 0, 40 / 180 * pi, 0],
    "base_to_world_pose": [0, 0, 1148, 0, 0, 0],
}


class PointCloudGenerator:
    def __init__(
        self,
        camera_intrinsics=None,
        camera_extrinsics=None,
        save_point_cloud=False,
        save_path="point_cloud.npy",
        denoise=True,
        denoise_neighbors=10,
        denoise_std_ratio=5.0,
        use_dbscan=False,
        dbscan_eps=150.0,
        dbscan_min_points=60,
        depth_band_tolerance=60.0,
    ):
        camera_intrinsics = camera_intrinsics or DEFAULT_CAMERA_INTRINSICS
        camera_extrinsics = camera_extrinsics or DEFAULT_CAMERA_EXTRINSICS
        self.fx = camera_intrinsics["fx"]
        self.fy = camera_intrinsics["fy"]
        self.cx = camera_intrinsics["cx"]
        self.cy = camera_intrinsics["cy"]
        self.tcp_pose = camera_extrinsics.get("tcp_pose", [0, 0, 0, 0, 0, 0])
        self.camera_to_tcp_pose = camera_extrinsics.get("camera_to_tcp_pose", [0, 0, 100, 0, 0, 0])
        self.base_to_world_pose = camera_extrinsics.get("base_to_world_pose", [0, 0, 0, 0, 0, 0])
        self.save_point_cloud = save_point_cloud
        self.save_path = save_path
        self.denoise = denoise
        self.denoise_neighbors = denoise_neighbors
        self.denoise_std_ratio = denoise_std_ratio
        self.use_dbscan = use_dbscan
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.depth_band_tolerance = float(depth_band_tolerance)
        self.generated_point_cloud = None

    @staticmethod
    def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = np.asarray(points, dtype=np.float64)
        if colors is not None:
            colors = np.asarray(colors)
            if colors.ndim != 2 or colors.shape[0] != points.shape[0] or colors.shape[1] < 3:
                colors = None
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {points.shape[0]}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            if colors is not None:
                handle.write("property uchar red\n")
                handle.write("property uchar green\n")
                handle.write("property uchar blue\n")
            handle.write("end_header\n")
            if colors is None:
                for x, y, z in points:
                    handle.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
            else:
                for point, color in zip(points, colors):
                    b, g, r = np.clip(color[:3], 0, 255).astype(np.uint8)
                    handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {int(r)} {int(g)} {int(b)}\n")

    @staticmethod
    def rotation_matrix(r_x, r_y, r_z):
        rx = np.array([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
        ry = np.array([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0], [-np.sin(r_y), 0, np.cos(r_y)]])
        rz = np.array([[np.cos(r_z), -np.sin(r_z), 0], [np.sin(r_z), np.cos(r_z), 0], [0, 0, 1]])
        return rz @ ry @ rx

    @staticmethod
    def pose_to_matrix(pose):
        x, y, z, rx, ry, rz = pose
        transform = np.eye(4)
        transform[:3, :3] = PointCloudGenerator.rotation_matrix(rx, ry, rz)
        transform[:3, 3] = [x, y, z]
        return transform

    @staticmethod
    def downsample_image(color_img, depth_img, scale_percent=1.0):
        if scale_percent == 1.0:
            return color_img.copy(), depth_img.copy()
        interval = max(1, int(1 / np.sqrt(scale_percent)))
        resized_color = np.zeros_like(color_img)
        resized_depth = np.zeros_like(depth_img)
        for row in range(0, color_img.shape[0], interval):
            for col in range(0, color_img.shape[1], interval):
                resized_color[row, col] = color_img[row, col]
                resized_depth[row, col] = depth_img[row, col]
        return resized_color, resized_depth

    def refine_mask_by_depth(self, depth_img, mask):
        import cv2

        if mask.shape[:2] != depth_img.shape[:2]:
            mask = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask > 0).astype(np.uint8)
        valid_pixels = (mask_binary > 0) & (depth_img > 0)
        if not np.any(valid_pixels):
            return np.zeros_like(mask_binary, dtype=np.uint8)
        depth_values = depth_img[valid_pixels].astype(np.float32)
        median_depth = float(np.median(depth_values))
        refined_mask = mask_binary & (np.abs(depth_img.astype(np.float32) - median_depth) <= self.depth_band_tolerance).astype(np.uint8)
        return refined_mask.astype(np.uint8) if np.any(refined_mask) else mask_binary

    def remove_outliers(self, points, colors=None):
        if not self.denoise or points.shape[0] < max(2, self.denoise_neighbors):
            return points, colors
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=self.denoise_neighbors + 1).fit(points)
            distances, _ = nbrs.kneighbors(points)
            distances = distances[:, 1:]
            mean_dist = distances.mean(axis=1)
            mask = mean_dist <= mean_dist.mean() + self.denoise_std_ratio * mean_dist.std()
            points = points[mask]
            colors = colors[mask] if colors is not None else None
            if self.use_dbscan and len(points) > 0:
                clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_points).fit(points)
                labels = clustering.labels_
                counts = np.bincount(labels[labels >= 0]) if np.any(labels >= 0) else np.array([])
                if len(counts) > 0:
                    keep = labels == int(np.argmax(counts))
                    points = points[keep]
                    colors = colors[keep] if colors is not None else None
        except Exception:
            pass
        return points, colors

    def generate_point_cloud(self, color_image_aligned, depth_image_aligned, mask=None, downsample_scale=1.0, target_coordinate_system="world"):
        if color_image_aligned is None or depth_image_aligned is None:
            return {"state": "fail", "info": "Invalid input image", "point_cloud": None}
        try:
            color_img = color_image_aligned.copy()
            depth_img = depth_image_aligned.copy()
            if depth_img.ndim == 3 and depth_img.shape[2] == 1:
                depth_img = depth_img.squeeze(axis=2)
            if mask is not None:
                refined_mask = self.refine_mask_by_depth(depth_img, mask)
                if refined_mask.shape[:2] != color_img.shape[:2]:
                    import cv2

                    refined_mask = cv2.resize(refined_mask, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                color_img = color_img * np.stack([refined_mask] * 3, axis=-1)
                depth_img = depth_img * refined_mask
                if not np.any(depth_img > 0):
                    return {"state": "fail", "info": "No valid depth in mask", "point_cloud": None}

            color_down, depth_down = self.downsample_image(color_img, depth_img, scale_percent=downsample_scale)
            fx_f = self.fx * downsample_scale
            fy_f = self.fy * downsample_scale
            cx_f = self.cx * downsample_scale
            cy_f = self.cy * downsample_scale
            valid_mask = depth_down > 0
            if not np.any(valid_mask):
                return {"state": "fail", "info": "No valid depth", "point_cloud": None}

            valid_depth = depth_down[valid_mask].astype(np.float32)
            valid_color = color_down[valid_mask]
            v, u = np.where(valid_mask)
            x_cam = (u - cx_f) * valid_depth / fx_f
            y_cam = (v - cy_f) * valid_depth / fy_f
            z_cam = valid_depth
            points_cam = np.stack((x_cam, y_cam, z_cam), axis=1)

            if target_coordinate_system == "camera":
                points_target = points_cam
            else:
                cam_to_tcp = self.pose_to_matrix(self.camera_to_tcp_pose)
                points_tcp = points_cam @ cam_to_tcp[:3, :3].T + cam_to_tcp[:3, 3]
                tcp_to_base = self.pose_to_matrix(self.tcp_pose)
                points_base = points_tcp @ tcp_to_base[:3, :3].T + tcp_to_base[:3, 3]
                if target_coordinate_system == "base":
                    points_target = points_base
                else:
                    base_to_world = self.pose_to_matrix(self.base_to_world_pose)
                    points_target = points_base @ base_to_world[:3, :3].T + base_to_world[:3, 3]

            points_target, valid_color = self.remove_outliers(points_target, valid_color)
            if points_target is None or len(points_target) == 0:
                return {"state": "fail", "info": "No points after filtering", "point_cloud": None}

            if self.save_point_cloud and self.save_path:
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(self.save_path, points_target)
                self.save_ply(Path(self.save_path).with_suffix(".ply"), points_target, valid_color)

            self.generated_point_cloud = points_target
            x_min, y_min, z_min = np.min(points_target, axis=0)
            x_max, y_max, z_max = np.max(points_target, axis=0)
            center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0])
            return {
                "state": "success",
                "info": f"point cloud generated: {len(points_target)} points",
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "x_min": float(x_min),
                "y_min": float(y_min),
                "z_min": float(z_min),
                "x_max": float(x_max),
                "y_max": float(y_max),
                "z_max": float(z_max),
                "point_count": int(len(points_target)),
                "point_cloud": points_target,
            }
        except Exception as exc:
            return {"state": "fail", "info": f"Point cloud generation failed: {exc}", "point_cloud": None}
