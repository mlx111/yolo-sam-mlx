from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class CameraRGBDSaveSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("camera_rgbd_save", recovery_steps.camera_image)


def load_skill() -> CameraRGBDSaveSkill:
    return CameraRGBDSaveSkill()
