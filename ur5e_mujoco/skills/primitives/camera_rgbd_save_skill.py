from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class CameraRGBDSaveSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("camera_rgbd_save", recovery_steps.camera_image)


def load_skill() -> CameraRGBDSaveSkill:
    return CameraRGBDSaveSkill()

