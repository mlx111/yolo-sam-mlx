from __future__ import annotations

from pathlib import Path

from skills.base.head_camera_grounded_sam2_pose_skill import R1ProHeadCameraGroundedSAM2PoseSkill


def load_skill(path: str | Path | None = None) -> R1ProHeadCameraGroundedSAM2PoseSkill:
    return R1ProHeadCameraGroundedSAM2PoseSkill()

