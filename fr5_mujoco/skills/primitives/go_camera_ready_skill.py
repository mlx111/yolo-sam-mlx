from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class GoCameraReadySkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("go_camera_ready", recovery_steps.execute_camera_ready)


def load_skill() -> GoCameraReadySkill:
    return GoCameraReadySkill()
