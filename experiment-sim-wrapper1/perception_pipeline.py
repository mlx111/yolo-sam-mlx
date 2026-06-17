"""
Perception pipeline: render → YOLO/SAM2 → point cloud → pose estimation.

Wraps existing code from the root project (cv_proc, object_pose_runtime, etc.)
into a unified interface.  All positions returned are from perception only —
never MuJoCo ground-truth.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # → .../YOLO_World-SAM-GraspNet
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "Grounded-SAM-2"))

from cv_proc import segment_image_ground
from build_runtime_scene_from_sim_camera import (
    _camera_config,
    _backproject_masked_depth,
    _cv_points_to_world,
)
import object_pose_runtime


# ---------------------------------------------------------------------------
# Lightweight data object returned by the pipeline
# ---------------------------------------------------------------------------

class PerceivedScene:
    """Everything returned by the perception pipeline.

    All fields come from camera images + YOLO/SAM2 + point-cloud processing,
    *not* from MuJoCo internals.
    """
    def __init__(
        self,
        apple_pos: Optional[np.ndarray] = None,
        apple_quat: Optional[np.ndarray] = None,
        confidence: float = 0.0,
        mask_nonzero: int = 0,
        detection_ok: bool = False,
    ):
        self.apple_pos = apple_pos          # (3,) world-frame position
        self.apple_quat = apple_quat        # (4,) wxyz quaternion or None
        self.confidence = confidence
        self.mask_nonzero = mask_nonzero
        self.detection_ok = detection_ok

    def __repr__(self) -> str:
        pos = f"({self.apple_pos[0]:.4f},{self.apple_pos[1]:.4f},{self.apple_pos[2]:.4f})" if self.apple_pos is not None else "None"
        return f"PerceivedScene(pos={pos}, conf={self.confidence:.3f}, ok={self.detection_ok})"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class PerceptionPipeline:
    """Render MuJoCo camera → YOLO-World + SAM2 → point cloud → 6D pose.

    Usage
    -----
    pipe = PerceptionPipeline(model, data)
    scene = pipe.detect()               # quick: position only
    scene = pipe.detect(full_pose=True) # includes quaternion estimate
    """

    CAMERA_NAME = "cam1"
    RENDER_HEIGHT = 640
    RENDER_WIDTH = 480
    DEPTH_TRUNC_M = 6.0
    MASK_ERODE_PIXELS = 1

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(
            model, height=self.RENDER_HEIGHT, width=self.RENDER_WIDTH
        )

    # ── public API ────────────────────────────────────────────────────

    def detect(self, target_class: str = "apple",
               full_pose: bool = False,
               work_dir: str | Path = "/tmp/perception") -> PerceivedScene:
        """Run the full perception pipeline and return the perceived scene.

        Parameters
        ----------
        target_class : str
            Object to detect (e.g. "apple", "pear").
        full_pose : bool
            If True, also estimate 6D orientation (slower, requires calibration).
        work_dir : str | Path
            Directory for temporary image files.

        Returns
        -------
        PerceivedScene with perceived (not ground-truth) object state.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Render RGB + depth -------------------------------------------------
        rgb_path = work_dir / "color_img1.jpg"
        depth_path = work_dir / "depth.npy"

        self._render_rgb_depth(rgb_path, depth_path)

        # 2. YOLO-World + SAM2 → mask -------------------------------------------
        mask = segment_image_ground(
            str(rgb_path),
            target_class,
            output_mask=str(work_dir / "mask.png"),
        )
        mask = np.asarray(mask).squeeze() > 0  # force 2D bool
        if np.count_nonzero(mask) == 0:
            print("[PerceptionPipeline] detection returned empty mask")
            return PerceivedScene(detection_ok=False)

        # 2b. Sanity: mask should overlap with foreground (apple-colored pixels).
        # This rejects hallucinated masks that pass YOLO+SAM2 but have zero overlap
        # with the actual object color (common when the robot arm enters the FOV).
        rgb_bgr = cv2.imread(str(rgb_path))
        if rgb_bgr is not None:
            hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            fg_color = ((h < 14) | (h > 165)) & (s > 45) & (v > 35) if target_class == "apple" else ((h > 12) & (h < 50) & (s > 40) & (v > 45))
            overlap = int(np.count_nonzero(mask & fg_color))
            mask_px = int(np.count_nonzero(mask))
            if mask_px > 100 and overlap / mask_px < 0.05:
                print(f"[PerceptionPipeline] mask-foreground overlap {overlap}/{mask_px} < 5% — rejecting")
                return PerceivedScene(detection_ok=False)

        # 3. Camera geometry from MuJoCo ----------------------------------------
        cam_config = _camera_config(
            self.model, self.data, self.CAMERA_NAME,
            self.RENDER_WIDTH, self.RENDER_HEIGHT,
        )
        intrinsics = cam_config["intrinsics"]
        rotation_world_from_cv = np.asarray(
            cam_config["rotation_matrix_world_from_cam"], dtype=np.float64
        )
        translation = np.asarray(cam_config["translation_mj"], dtype=np.float64)

        # 4. Backproject masked depth → world-frame point cloud -----------------
        depth = np.load(depth_path)
        points_cv = _backproject_masked_depth(depth, mask, intrinsics)
        if len(points_cv) < 8:
            print("[PerceptionPipeline] too few valid depth points")
            return PerceivedScene(detection_ok=False)

        # Sanity: mask-to-depth ratio — a good detection should have >5% of mask
        # pixels producing valid 3D points (rejects detections on reflective/shiny
        # surfaces where the depth sensor returns no data).
        mask_pixels = np.count_nonzero(mask)
        depth_ratio = len(points_cv) / max(mask_pixels, 1)
        if depth_ratio < 0.05:
            print(f"[PerceptionPipeline] mask-depth ratio {depth_ratio:.3f} < 0.05 — rejecting")
            return PerceivedScene(detection_ok=False)

        points_world = _cv_points_to_world(
            points_cv, rotation_world_from_cv, translation
        )

        # 4b. Sanity: point cloud should be spatially compact
        if len(points_world) > 20:
            z_std = float(np.std(points_world[:, 2]))
            if z_std > 0.12:
                print(f"[PerceptionPipeline] point cloud Z std {z_std:.4f}m > 0.12m — rejecting")
                return PerceivedScene(detection_ok=False)

        # 5. Position from trimmed-mean of world points -------------------------
        apple_pos = self._robust_position(points_world)
        confidence = float(min(1.0, len(points_world) / 5000))

        # 6. Optional 6D orientation --------------------------------------------
        apple_quat = None
        if full_pose:
            apple_quat = self._estimate_orientation(
                points_world, target_class, work_dir, cam_config
            )

        return PerceivedScene(
            apple_pos=apple_pos,
            apple_quat=apple_quat,
            confidence=confidence,
            mask_nonzero=int(np.count_nonzero(mask)),
            detection_ok=True,
        )

    # ── internal helpers ──────────────────────────────────────────────

    def _render_rgb_depth(self, rgb_path: Path, depth_path: Path) -> None:
        """Render RGB and depth from MuJoCo, save to disk."""
        # RGB
        self.renderer.update_scene(self.data, camera=self.CAMERA_NAME)
        rgb = self.renderer.render()
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_path), rgb_bgr)

        # Depth
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA_NAME)
        depth = self.renderer.render()
        np.save(str(depth_path), depth)

    @staticmethod
    def _robust_position(points_world: np.ndarray) -> np.ndarray:
        """Trimmed-mean position estimate (same logic as build_runtime_scene)."""
        if len(points_world) < 8:
            return np.asarray(points_world.mean(axis=0), dtype=np.float64)
        lo = np.quantile(points_world, 0.02, axis=0)
        hi = np.quantile(points_world, 0.98, axis=0)
        keep = np.all((points_world >= lo) & (points_world <= hi), axis=1)
        trimmed = points_world[keep]
        if len(trimmed) < 8:
            trimmed = points_world
        return np.asarray(trimmed.mean(axis=0), dtype=np.float64)

    def _estimate_orientation(
        self, points_world: np.ndarray,
        target_class: str,
        work_dir: Path,
        cam_config: dict,
    ) -> np.ndarray:
        """Estimate object orientation using object_pose_runtime.

        This requires a calibration JSON — we build a minimal one on the fly.
        """
        # Save point cloud for object_pose_runtime to consume
        np.save(work_dir / "raw_left_apple.npy", points_world)

        # Build minimal calibration
        calib = {
            "camera_poses": {
                "left": {
                    "translation_mj": [float(v) for v in cam_config["translation_mj"]],
                    "quat_wxyz": [float(v) for v in cam_config["quat_wxyz_mj"]],
                    "rotation_matrix_mj_from_cam": cam_config["rotation_matrix_mj_from_cam"].tolist(),
                    "rotation_matrix_world_from_cam": cam_config["rotation_matrix_world_from_cam"].tolist(),
                    "intrinsics": {k: float(v) if k not in {"width", "height"} else int(v)
                                   for k, v in cam_config["intrinsics"].items()},
                    "fovy_deg": float(cam_config["fovy_deg"]),
                }
            },
            "object_positions": {},
        }
        calib_path = work_dir / "calibration.json"
        import json
        calib_path.write_text(json.dumps(calib, indent=2))

        # Patch module-level intrinsics (required by object_pose_runtime)
        left_intr = cam_config["intrinsics"]
        object_pose_runtime.LEFT_INTRINSICS.clear()
        object_pose_runtime.LEFT_INTRINSICS.update(left_intr)

        quats = object_pose_runtime.estimate_runtime_object_quats(
            camera="left",
            calibration_path=str(calib_path),
            pear_strategy="v0_legacy",
        )
        if target_class in quats:
            return np.asarray(quats[target_class], dtype=np.float64)
        return None

    def close(self):
        self.renderer.close()
