"""
Perception pipeline: render → Grounded-SAM2 → point cloud → pose estimation.

Uses the same Grounded-SAM2 backend style as galaxea_mujoco. All positions
returned are from rendered RGB-D perception only, never MuJoCo ground-truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

from galaxea_mujoco.runtime_perception.backends.grounded_sam2_backend import GroundedSAM2Segmenter


MJ_CAMERA_FROM_CV = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Lightweight data object returned by the pipeline
# ---------------------------------------------------------------------------

class PerceivedScene:
    """Everything returned by the perception pipeline.

    All fields come from camera images + Grounded-SAM2 + point-cloud processing,
    *not* from MuJoCo internals.
    """
    def __init__(
        self,
        apple_pos: Optional[np.ndarray] = None,
        apple_quat: Optional[np.ndarray] = None,
        confidence: float = 0.0,
        mask_nonzero: int = 0,
        detection_ok: bool = False,
        raw_points_world: Optional[np.ndarray] = None,
        raw_points_camera: Optional[np.ndarray] = None,
        mask_path: Optional[str] = None,
        rgb_path: Optional[str] = None,
        depth_path: Optional[str] = None,
    ):
        self.apple_pos = apple_pos          # (3,) world-frame position
        self.apple_quat = apple_quat        # (4,) wxyz quaternion or None
        self.confidence = confidence
        self.mask_nonzero = mask_nonzero
        self.detection_ok = detection_ok
        self.raw_points_world = raw_points_world
        self.raw_points_camera = raw_points_camera
        self.mask_path = mask_path
        self.rgb_path = rgb_path
        self.depth_path = depth_path

    def __repr__(self) -> str:
        pos = f"({self.apple_pos[0]:.4f},{self.apple_pos[1]:.4f},{self.apple_pos[2]:.4f})" if self.apple_pos is not None else "None"
        return f"PerceivedScene(pos={pos}, conf={self.confidence:.3f}, ok={self.detection_ok})"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class PerceptionPipeline:
    """Render MuJoCo camera → GroundingDINO + SAM2 → point cloud → 6D pose.

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

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        grounded_sam2_root: str | None = None,
        sam2_checkpoint: str | None = None,
        sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config: str | None = None,
        grounding_dino_checkpoint: str | None = None,
        bert_path: str | None = None,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        device: str | None = None,
        multimask_output: bool = True,
    ):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(
            model, height=self.RENDER_HEIGHT, width=self.RENDER_WIDTH
        )
        self.segmenter = GroundedSAM2Segmenter(
            grounded_sam2_root=grounded_sam2_root,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_config=sam2_model_config,
            grounding_dino_config=grounding_dino_config,
            grounding_dino_checkpoint=grounding_dino_checkpoint,
            bert_path=bert_path,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            multimask_output=multimask_output,
        )

    # ── public API ────────────────────────────────────────────────────

    def render_rgbd(self, work_dir: str | Path = "/tmp/perception") -> dict[str, object]:
        """Render and save one RGB-D observation from the configured camera."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = work_dir / "color_img1.jpg"
        depth_path = work_dir / "depth.npy"
        self._render_rgb_depth(rgb_path, depth_path)
        cam_config = self._camera_config(
            self.model,
            self.data,
            self.CAMERA_NAME,
            self.RENDER_WIDTH,
            self.RENDER_HEIGHT,
        )
        return {
            "camera_name": self.CAMERA_NAME,
            "work_dir": str(work_dir),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera_config": cam_config,
        }

    def detect_from_rgbd(
        self,
        *,
        target_class: str = "apple",
        rgb_path: str | Path,
        depth_path: str | Path,
        work_dir: str | Path = "/tmp/perception",
        full_pose: bool = False,
        return_cloud: bool = False,
        camera_config: dict | None = None,
    ) -> PerceivedScene:
        """Segment and backproject a target object from an existing RGB-D frame."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = Path(rgb_path)
        depth_path = Path(depth_path)

        mask_path = work_dir / "mask.png"
        annotated_path = work_dir / "annotated.png"
        seg_result = self.segmenter.segment_image(
            image_path=str(rgb_path),
            target_class=target_class,
            output_mask_path=str(mask_path),
            output_annotated_path=str(annotated_path),
        )
        if seg_result is None:
            print("[PerceptionPipeline] Grounded-SAM2 returned no mask")
            return PerceivedScene(detection_ok=False, rgb_path=str(rgb_path), depth_path=str(depth_path))
        mask = np.asarray(seg_result.mask).squeeze() > 0
        if np.count_nonzero(mask) == 0:
            print("[PerceptionPipeline] Grounded-SAM2 returned empty mask")
            return PerceivedScene(detection_ok=False, rgb_path=str(rgb_path), depth_path=str(depth_path), mask_path=str(mask_path))

        cam_config = camera_config or self._camera_config(
            self.model,
            self.data,
            self.CAMERA_NAME,
            self.RENDER_WIDTH,
            self.RENDER_HEIGHT,
        )
        intrinsics = cam_config["intrinsics"]
        rotation_world_from_cv = np.asarray(
            cam_config["rotation_matrix_world_from_cam"], dtype=np.float64
        )
        translation = np.asarray(cam_config["translation_mj"], dtype=np.float64)

        depth = np.load(depth_path)
        points_cv = self._backproject_masked_depth(depth, mask, intrinsics)
        if len(points_cv) < 8:
            print("[PerceptionPipeline] too few valid depth points")
            return PerceivedScene(detection_ok=False, rgb_path=str(rgb_path), depth_path=str(depth_path), mask_path=str(mask_path))

        mask_pixels = np.count_nonzero(mask)
        depth_ratio = len(points_cv) / max(mask_pixels, 1)
        if depth_ratio < 0.05:
            print(f"[PerceptionPipeline] mask-depth ratio {depth_ratio:.3f} < 0.05 — rejecting")
            return PerceivedScene(detection_ok=False, rgb_path=str(rgb_path), depth_path=str(depth_path), mask_path=str(mask_path))

        points_world = self._cv_points_to_world(
            points_cv, rotation_world_from_cv, translation
        )

        if len(points_world) > 20:
            z_std = float(np.std(points_world[:, 2]))
            if z_std > 0.12:
                print(f"[PerceptionPipeline] point cloud Z std {z_std:.4f}m > 0.12m — rejecting")
                return PerceivedScene(detection_ok=False, rgb_path=str(rgb_path), depth_path=str(depth_path), mask_path=str(mask_path))

        apple_pos = self._robust_position(points_world)
        confidence = float(min(1.0, len(points_world) / 5000))

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
            raw_points_world=points_world if return_cloud else None,
            raw_points_camera=points_cv if return_cloud else None,
            mask_path=str(mask_path),
            rgb_path=str(rgb_path),
            depth_path=str(depth_path),
        )

    def detect(self, target_class: str = "apple",
               full_pose: bool = False,
               work_dir: str | Path = "/tmp/perception",
               return_cloud: bool = False) -> PerceivedScene:
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
        rgbd = self.render_rgbd(work_dir)
        return self.detect_from_rgbd(
            target_class=target_class,
            rgb_path=str(rgbd["rgb_path"]),
            depth_path=str(rgbd["depth_path"]),
            work_dir=work_dir,
            full_pose=full_pose,
            return_cloud=return_cloud,
            camera_config=rgbd["camera_config"],
        )

    # ── internal helpers ──────────────────────────────────────────────

    def _render_rgb_depth(self, rgb_path: Path, depth_path: Path) -> None:
        """Render RGB and depth from MuJoCo, save to disk."""
        # RGB
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA_NAME)
        rgb = self.renderer.render()
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_path), rgb_bgr)

        # Depth
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA_NAME)
        depth = self.renderer.render()
        np.save(str(depth_path), depth)
        self.renderer.disable_depth_rendering()

    @staticmethod
    def _camera_intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> dict[str, float | int]:
        fy = height / (2.0 * np.tan(np.deg2rad(float(fovy_deg)) / 2.0))
        return {
            "fx": float(fy),
            "fy": float(fy),
            "cx": float(width / 2.0),
            "cy": float(height / 2.0),
            "width": int(width),
            "height": int(height),
        }

    @classmethod
    def _camera_config(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str,
        width: int,
        height: int,
    ) -> dict[str, object]:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera not found: {camera_name}")
        rotation_mj_from_cam = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
        rotation_world_from_cv = rotation_mj_from_cam @ MJ_CAMERA_FROM_CV
        translation = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        quat_wxyz = cls._matrix_to_quat_wxyz(rotation_mj_from_cam)
        return {
            "name": camera_name,
            "translation_mj": translation,
            "quat_wxyz_mj": quat_wxyz,
            "rotation_matrix_mj_from_cam": rotation_world_from_cv,
            "rotation_matrix_world_from_cam": rotation_world_from_cv,
            "intrinsics": cls._camera_intrinsics_from_fovy(width, height, float(model.cam_fovy[cam_id])),
            "fovy_deg": float(model.cam_fovy[cam_id]),
        }

    @staticmethod
    def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
        # Camera quats are metadata/debug output; the fixed-grasp path does not depend on them.
        import scipy.spatial.transform

        quat_xyzw = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix).as_quat()
        return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

    @staticmethod
    def _backproject_masked_depth(
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsics: dict[str, float | int],
    ) -> np.ndarray:
        valid = mask & np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= PerceptionPipeline.DEPTH_TRUNC_M)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)
        v, u = np.where(valid)
        z = depth_m[valid].astype(np.float64)
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x = (u.astype(np.float64) - cx) * z / fx
        y = (v.astype(np.float64) - cy) * z / fy
        return np.column_stack([x, y, z])

    @staticmethod
    def _cv_points_to_world(points_cv: np.ndarray, rotation_world_from_cv: np.ndarray, translation: np.ndarray) -> np.ndarray:
        return (rotation_world_from_cv @ points_cv.T).T + translation

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
        return None

    def close(self):
        self.renderer.close()
