from __future__ import annotations

from pathlib import Path

from skills.base.head_camera_rgbd_save_skill import R1ProHeadCameraRGBDSaveSkill


def load_skill(path: str | Path | None = None) -> R1ProHeadCameraRGBDSaveSkill:
    return R1ProHeadCameraRGBDSaveSkill()

