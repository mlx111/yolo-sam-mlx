from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class DetectObjectPoseSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("detect_object_pose", recovery_steps.detect_object)


def load_skill() -> DetectObjectPoseSkill:
    return DetectObjectPoseSkill()

