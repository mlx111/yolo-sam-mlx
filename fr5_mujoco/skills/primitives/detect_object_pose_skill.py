from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class DetectObjectPoseSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("detect_object_pose", recovery_steps.detect_object)


def load_skill() -> DetectObjectPoseSkill:
    return DetectObjectPoseSkill()
