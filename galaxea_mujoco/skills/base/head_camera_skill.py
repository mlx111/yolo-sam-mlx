from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class HeadCameraFrame:
    camera_name: str
    rgb: np.ndarray
    depth: np.ndarray | None
    camera_position: np.ndarray
    camera_xmat: np.ndarray


class R1ProHeadCameraSkill:
    """Read RGB and depth images from the externally mounted head RGB-D camera."""

    def __init__(self, camera_name: str = "external_head_rgbd_camera", width: int = 320, height: int = 240):
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)

    def capture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        width: int | None = None,
        height: int | None = None,
        include_depth: bool = True,
    ) -> HeadCameraFrame:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {self.camera_name}")

        image_width = self.width if width is None else int(width)
        image_height = self.height if height is None else int(height)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=image_height, width=image_width)
        try:
            renderer.update_scene(data, camera=self.camera_name)
            rgb = renderer.render().copy()
            depth = None
            if include_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=self.camera_name)
                depth = renderer.render().copy()
        finally:
            renderer.close()

        return HeadCameraFrame(
            camera_name=self.camera_name,
            rgb=rgb,
            depth=depth,
            camera_position=data.cam_xpos[camera_id].copy(),
            camera_xmat=data.cam_xmat[camera_id].reshape(3, 3).copy(),
        )

    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> HeadCameraFrame:
        return self.capture(
            model,
            data,
            width=params.get("width"),
            height=params.get("height"),
            include_depth=bool(params.get("include_depth", True)),
        )


def load_skill(camera_name: str = "external_head_rgbd_camera", width: int = 320, height: int = 240) -> R1ProHeadCameraSkill:
    return R1ProHeadCameraSkill(camera_name=camera_name, width=width, height=height)
