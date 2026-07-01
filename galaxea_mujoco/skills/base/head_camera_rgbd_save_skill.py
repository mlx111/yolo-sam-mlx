from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


@dataclass(frozen=True)
class HeadCameraRGBDSaveResult:
    success: bool
    camera_name: str
    width: int
    height: int
    rgb_path: str
    depth_path: str
    depth_png_path: str
    metadata_path: str
    camera_position: list[float]
    camera_xmat: list[list[float]]
    intrinsics: dict[str, float | int]
    depth_min: float | None
    depth_max: float | None
    depth_median: float | None
    valid_depth_count: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProHeadCameraRGBDSaveSkill:
    """Capture RGB-D from a MuJoCo fixed camera and save images plus metadata."""

    def __init__(
        self,
        camera_name: str = "head_top_work_camera",
        width: int = 640,
        height: int = 480,
        output_dir: str | Path = "output/head_camera_rgbd",
    ):
        self.camera_name = str(camera_name)
        self.width = int(width)
        self.height = int(height)
        self.output_dir = Path(output_dir)

    def capture_and_save(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        camera_name: str | None = None,
        width: int | None = None,
        height: int | None = None,
        output_dir: str | Path | None = None,
        prefix: str = "head_top_work_camera",
    ) -> HeadCameraRGBDSaveResult:
        name = self.camera_name if camera_name is None else str(camera_name)
        image_width = self.width if width is None else int(width)
        image_height = self.height if height is None else int(height)
        out_dir = self.output_dir if output_dir is None else Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {name}")

        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=image_height, width=image_width)
        try:
            renderer.update_scene(data, camera=name)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=name)
            depth = renderer.render().copy().astype(np.float32)
        finally:
            renderer.close()

        rgb_path = out_dir / f"{prefix}_rgb.png"
        depth_path = out_dir / f"{prefix}_depth.npy"
        depth_png_path = out_dir / f"{prefix}_depth_meters.png"
        metadata_path = out_dir / f"{prefix}_metadata.json"

        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth)
        self._save_depth_png(depth, depth_png_path)

        camera_position = data.cam_xpos[camera_id].copy()
        camera_xmat = data.cam_xmat[camera_id].reshape(3, 3).copy()
        intrinsics = self._camera_intrinsics(model, camera_id, image_width, image_height)
        depth_stats = self._depth_stats(depth)

        result = HeadCameraRGBDSaveResult(
            success=True,
            camera_name=name,
            width=image_width,
            height=image_height,
            rgb_path=str(rgb_path),
            depth_path=str(depth_path),
            depth_png_path=str(depth_png_path),
            metadata_path=str(metadata_path),
            camera_position=np.round(camera_position, 9).tolist(),
            camera_xmat=np.round(camera_xmat, 9).tolist(),
            intrinsics=intrinsics,
            depth_min=depth_stats["min"],
            depth_max=depth_stats["max"],
            depth_median=depth_stats["median"],
            valid_depth_count=int(depth_stats["valid_count"]),
            message="RGB-D capture saved",
        )
        metadata_path.write_text(self._json_dumps(result.to_dict()), encoding="utf-8")
        return result

    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> HeadCameraRGBDSaveResult:
        runtime_tmp_dir = params.get("_runtime_tmp_dir")
        output_dir = params.get("output_dir")
        if output_dir is None and runtime_tmp_dir:
            output_dir = Path(str(runtime_tmp_dir)) / "head_camera_rgbd"
        return self.capture_and_save(
            model,
            data,
            camera_name=params.get("camera_name"),
            width=params.get("width"),
            height=params.get("height"),
            output_dir=output_dir,
            prefix=str(params.get("prefix", "head_top")),
        )

    @staticmethod
    def _camera_intrinsics(model: mujoco.MjModel, camera_id: int, width: int, height: int) -> dict[str, float | int]:
        fovy = float(model.cam_fovy[camera_id])
        fy = 0.5 * float(height) / np.tan(0.5 * np.deg2rad(fovy))
        fx = fy
        cx = (float(width) - 1.0) * 0.5
        cy = (float(height) - 1.0) * 0.5
        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width),
            "height": int(height),
            "fovy": fovy,
        }

    @staticmethod
    def _depth_stats(depth: np.ndarray) -> dict[str, float | int | None]:
        valid = np.asarray(depth, dtype=np.float32)
        valid = valid[np.isfinite(valid) & (valid > 0.0)]
        if valid.size == 0:
            return {"min": None, "max": None, "median": None, "valid_count": 0}
        return {
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "median": float(np.median(valid)),
            "valid_count": int(valid.size),
        }

    @staticmethod
    def _save_depth_png(depth: np.ndarray, path: Path) -> None:
        depth_millimeters = np.asarray(depth, dtype=np.float32) * 1000.0
        depth_millimeters = np.nan_to_num(depth_millimeters, nan=0.0, posinf=0.0, neginf=0.0)
        depth_uint16 = np.clip(depth_millimeters, 0.0, 65535.0).astype(np.uint16)
        cv2.imwrite(str(path), depth_uint16)

    @staticmethod
    def _json_dumps(payload: dict[str, Any]) -> str:
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2)


def load_skill(
    camera_name: str = "head_top_work_camera",
    width: int = 640,
    height: int = 480,
    output_dir: str | Path = "output/head_camera_rgbd",
) -> R1ProHeadCameraRGBDSaveSkill:
    return R1ProHeadCameraRGBDSaveSkill(camera_name=camera_name, width=width, height=height, output_dir=output_dir)
